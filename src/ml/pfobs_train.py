# -*- coding: utf-8 -*-
"""DDP training of one PF-observability ``(arm, seed)`` run.

``train_one`` is called by the matrix runner (Task 6), which owns the process
group: one ``torchrun`` job initializes it once and then calls ``train_one``
once per (arm, seed) with the same :class:`DistEnv`. ``train_one`` therefore
never initializes or destroys the group, and it enters a barrier before
returning so no rank can start the next run while rank 0 is still writing
artifacts.

The loop is the m3 loop of :func:`src.ml.train.train_m3_dcs` -- masked
target-standardized MSE, AdamW, linear warm-up + half-cosine decay, gradient
clipping, synchronized early stopping -- expressed as a *global* loss. Each
rank's local masked SSE is scaled by ``world_size / global_masked_count`` so
that after DDP's gradient averaging the step minimizes the masked MSE over the
whole global batch (the identity ``tests/test_pf_observability_ddp.py`` pins).
Model init is seeded identically on every rank; dropout streams are reseeded
per rank (``seed + 10_000 * rank``) only after the DDP construction.
"""
import copy
import dataclasses
import functools
import io
import json
import math
import os
import pathlib

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import DCSWindowDataset, pad_collate
from .dcs_features import load_dcs_config
from .models import ActSeqAttn
from .pf_observability import (
    ARMS, PFObsSeriesReader, load_split, run_fingerprint, series_mean_std,
)
from .pfobs_provenance import normalization_stats_sha256
from .target import N_OUT, target_mean_std
from ..utils import (
    PublicationIOError, durable_publish_bytes, read_regular_nofollow,
)

# The fixed base-model contract: the four arms are comparable only under the
# identical model, positional encoding and time axis, so those values (and the
# 21-column arm input) are pinned; everything else in hp.m3 stays free so
# smoke-scale runs can shrink the model and window.
INPUT_WIDTH = 21
BASE_CONTRACT = {
    "model": "ActSeqAttn",
    "pe": "rope_time",
    "time_axis": "native_gmag_bnd",
}


def validate_base_contract(cfg):
    """Reject any config that breaks the fixed base-model contract.

    ``model``/``pe``/``time_axis`` are required and must equal
    :data:`BASE_CONTRACT`; ``input_width`` and ``hp.m3.pe`` are checked only
    when present, so a partial contract dict still validates its shared keys.
    """
    missing = [k for k in BASE_CONTRACT if k not in cfg]
    if missing:
        raise ValueError(
            f"base contract: config is missing {missing}; required "
            f"{sorted(BASE_CONTRACT)}")
    for key, want in BASE_CONTRACT.items():
        if cfg[key] != want:
            raise ValueError(
                f"base contract: {key} must be {want!r}, got {cfg[key]!r} -- "
                "the four arms are only comparable under the fixed base model")
    width = cfg.get("input_width")
    if width is not None and int(width) != INPUT_WIDTH:
        raise ValueError(
            f"base contract: input_width must be {INPUT_WIDTH}, got {width}")
    hp_pe = cfg.get("hp", {}).get("m3", {}).get("pe")
    if hp_pe is not None and hp_pe != cfg["pe"]:
        raise ValueError(
            f"base contract: hp.m3.pe {hp_pe!r} contradicts pe {cfg['pe']!r}")
    return cfg


def local_batch_size(global_batch, world_size):
    """Per-rank batch: the global batch split evenly, never padded."""
    global_batch, world_size = int(global_batch), int(world_size)
    if world_size < 1 or global_batch < 1:
        raise ValueError(
            f"global batch {global_batch} and world size {world_size} "
            "must both be >= 1")
    if global_batch % world_size:
        raise ValueError(
            f"global batch {global_batch} is not divisible by world size "
            f"{world_size} -- DDP must not silently change the batch")
    return global_batch // world_size


# ── distributed setup (the matrix runner owns the group's lifetime) ──
@dataclasses.dataclass(frozen=True)
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def init_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        if world != 1:
            raise RuntimeError(
                "multi-rank training requires CUDA/NCCL")
        return DistEnv(0, 0, 1, torch.device("cpu"))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
    return DistEnv(
        rank, local, world, torch.device("cuda", local))


def dist_barrier(dist_env):
    """Barrier; a structural no-op at world 1 (no group exists to barrier on)."""
    if dist_env.world_size > 1:
        torch.distributed.barrier()


def dist_broadcast(tensor, dist_env, src=0):
    """In-place broadcast; a no-op at world 1 (rank 0 already holds the value)."""
    if dist_env.world_size > 1:
        torch.distributed.broadcast(tensor, src=src)
    return tensor


def teardown_dist(dist_env):
    """Destroy the process group; a no-op at world 1.

    Only the matrix runner calls this, once, after the whole requested matrix:
    ``train_one`` must never destroy a group it does not own.
    """
    if dist_env.world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def global_masked_loss(pred, target, mask, world_size):
    w = mask.unsqueeze(-1).float()
    numerator = (((pred - target) ** 2) * w).sum()
    denominator = w.sum().detach()
    if world_size > 1:
        torch.distributed.all_reduce(
            denominator, op=torch.distributed.ReduceOp.SUM)
    return (
        numerator * float(world_size)
        / denominator.clamp(min=1.0)
    )


class _ShardSampler(Sampler):
    """Rank ``r`` of ``w`` sees ``[r::w]`` -- disjoint, with NO padding.

    :class:`~torch.utils.data.distributed.DistributedSampler` pads the dataset
    length to a multiple of ``world_size`` (possibly duplicating windows) so
    every rank runs an equal number of steps -- right for training, wrong for
    validation, where a duplicated window would be counted twice in the global
    numerator/denominator. Validation therefore shards exactly: every window
    enters the global numerator/denominator exactly once.
    """

    def __init__(self, n, world_size, rank):
        self.n, self.world_size, self.rank = int(n), int(world_size), int(rank)

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world_size))

    def __len__(self):
        return max(0, (self.n - self.rank + self.world_size - 1)
                   // self.world_size)


def _finite_input_reader(reader):
    """Wrap an arm reader so non-finite channel values never reach the model.

    The Task-1 sidecar is deliberately an intersection, never a repair: rows
    missing any PF/Ip signal keep their raw (possibly non-finite) values and
    carry ``common_valid = False``. ``DCSWindowDataset`` zero-fills only the
    TARGET -- a NaN *input* would reach attention, and under causality one NaN
    key poisons every later output of its window; a zero loss weight
    multiplies NaN, it does not remove it. This applies ``read_series``'s zero
    policy to the sidecar channels (non-finite -> 0.0, mask unchanged). Valid
    rows are byte-identical: ``common_valid`` already guarantees they are
    finite.
    """
    def read(target_npz, cfg, ncm):
        A, mask = reader(target_npz, cfg, ncm)
        return np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), mask
    return read


def _bcast_array(values, dist_env, dtype):
    """Rank-0's 1-D stats array broadcast to every rank as a tensor."""
    t = torch.as_tensor(np.asarray(values, dtype), device=dist_env.device)
    return dist_broadcast(t, dist_env).cpu().numpy()


def _preflight_training_destination(path, *, label):
    path = pathlib.Path(path)
    if os.path.lexists(path):
        read_regular_nofollow(path, label=label)


def publish_training_artifacts(out_dir, artifact, validation_summary):
    """Publish ``m3.pt`` then its JSON identity, both immutable and durable."""
    out_dir = pathlib.Path(out_dir)
    artifact_path = out_dir / "m3.pt"
    validation_path = out_dir / "validation.json"
    # Reject every redirected/broken/nonregular destination before committing
    # either member, so the identity-last pair never starts from hostile state.
    _preflight_training_destination(
        artifact_path, label="Work 2 training artifact")
    _preflight_training_destination(
        validation_path, label="Work 2 validation identity")

    artifact_buffer = io.BytesIO()
    torch.save(artifact, artifact_buffer)
    artifact_bytes = artifact_buffer.getvalue()
    validation_bytes = json.dumps(
        validation_summary, indent=2, sort_keys=True).encode("utf-8")

    def validate_artifact(payload):
        try:
            loaded = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise PublicationIOError(
                f"cannot reload complete Work 2 m3.pt bytes: {exc}") from exc
        if (not isinstance(loaded, dict)
                or loaded.get("run_fingerprint")
                != artifact.get("run_fingerprint")
                or loaded.get("validation_summary") != validation_summary):
            raise PublicationIOError(
                "complete Work 2 m3.pt bytes fail fingerprint/identity validation")
        return loaded

    def validate_identity(payload):
        try:
            loaded = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PublicationIOError(
                f"cannot parse complete Work 2 validation identity: {exc}") \
                from exc
        if loaded != validation_summary:
            raise PublicationIOError(
                "Work 2 validation identity differs from the artifact summary")
        return loaded

    artifact_status = durable_publish_bytes(
        artifact_path,
        artifact_bytes,
        validator=validate_artifact,
        state_label="Work 2 training artifact",
    )
    identity_status = durable_publish_bytes(
        validation_path,
        validation_bytes,
        validator=validate_identity,
        state_label="Work 2 validation identity",
    )
    return artifact_status, identity_status


def _available_shots(npz_dir, sidecar_dir):
    """Shots present in BOTH the target dataset and the sidecar."""
    def stems(d):
        return {int(p.stem) for p in pathlib.Path(d).glob("*.npz")
                if p.stem.isdigit()}
    return stems(npz_dir) & stems(sidecar_dir)


def training_normalization(npz_dir, sidecar_dir, train_shots, arm):
    """The four train-only normalization arrays for one Work 2 arm."""
    reader = PFObsSeriesReader(pathlib.Path(sidecar_dir), arm)
    mean, std = series_mean_std(npz_dir, train_shots, reader)
    tgt_mean, tgt_std = target_mean_std(npz_dir, train_shots)
    return mean, std, tgt_mean, tgt_std


def train_one(npz_dir, sidecar_dir, config_path, split_path, arm, seed,
              out_dir, dist_env, epochs_override=None, normalization=None,
              run_fingerprint_payload=None,
              source_audit_identity_sha256="absent"):
    """Train one ``(arm, seed)``; rank 0 writes ``<out_dir>/m3.pt`` + JSON.

    Sequence: validate the base contract and the split (every listed shot must
    exist in both the target and sidecar datasets); build the arm reader;
    compute the 21-column input and 34-column target normalization from TRAIN
    shots only on rank 0 and broadcast them; build the train/validation window
    datasets (asserting the 21-column arm input); train under DDP with a
    global masked loss, a distributed train sampler (``set_epoch`` every
    epoch), a no-padding validation shard, a fixed AdamW warm-up + cosine
    schedule and synchronized early stopping; rank 0 atomically writes the
    artifact and ``validation.json``; every rank barriers before returning.

    ``seed`` seeds model init identically on every rank and (after the DDP
    construction) each rank's dropout stream as ``seed + 10_000 * rank``. It
    never touches the split. ``epochs_override`` replaces ``hp.m3.epochs``
    (smoke runs); the artifact's ``hp`` records the epochs that actually ran.
    Rank 0 stores one exact ``validation_summary`` in ``m3.pt`` and serializes
    that same mapping as ``validation.json``.
    """
    npz_dir = pathlib.Path(npz_dir)
    sidecar_dir = pathlib.Path(sidecar_dir)     # the reader does not coerce str
    cfg = validate_base_contract(load_dcs_config(config_path))
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    split = load_split(split_path, available_shots=_available_shots(
        npz_dir, sidecar_dir))
    reader = PFObsSeriesReader(sidecar_dir, arm)
    hpm = cfg["hp"]["m3"]
    pe = cfg["pe"]
    n_ep = int(hpm["epochs"]) if epochs_override is None else int(epochs_override)
    if n_ep < 1:
        raise ValueError(f"epochs must be >= 1, got {n_ep}")
    hp = dict(hpm)
    hp["epochs"] = n_ep                       # the artifact records what ran
    dev = dist_env.device

    # Train-shot-only normalization. The runner may precompute rank 0's arrays
    # for resume fingerprinting; direct callers retain the historical local
    # computation. Every rank receives byte-identical typed arrays.
    if dist_env.is_main:
        stats = (training_normalization(
            npz_dir, sidecar_dir, split.train, arm)
                 if normalization is None else normalization)
        mean, std, tgt_mean, tgt_std = stats
    else:
        mean = np.zeros(INPUT_WIDTH, np.float32)
        std = np.ones(INPUT_WIDTH, np.float32)
        tgt_mean = np.zeros(N_OUT, np.float64)
        tgt_std = np.ones(N_OUT, np.float64)
    mean = _bcast_array(mean, dist_env, np.float32)
    std = _bcast_array(std, dist_env, np.float32)
    tgt_mean = _bcast_array(tgt_mean, dist_env, np.float64)
    tgt_std = _bcast_array(tgt_std, dist_env, np.float64)
    normalization_hash = normalization_stats_sha256(
        mean, std, tgt_mean, tgt_std)
    if (run_fingerprint_payload is not None
            and run_fingerprint_payload.get("normalization_sha256")
            != normalization_hash):
        raise RuntimeError(
            "run_fingerprint normalization_sha256 does not match the arrays "
            "broadcast to training")

    kw = dict(cfg={}, ncm={}, mean=mean, std=std, pe=pe,
              d_model=hpm["d_model"], w=hpm["window"], ctx=hpm["ctx"],
              tgt_mean=tgt_mean, tgt_std=tgt_std,
              series_reader=_finite_input_reader(reader))
    ds_tr = DCSWindowDataset(npz_dir, split.train, **kw)
    ds_va = DCSWindowDataset(npz_dir, split.validation, **kw)
    assert ds_tr.n_act == INPUT_WIDTH, (
        f"arm {arm}: the fixed input is {INPUT_WIDTH} columns, "
        f"got {ds_tr.n_act}")

    lb = local_batch_size(hpm["batch"], dist_env.world_size)
    coll = functools.partial(pad_collate, w=hpm["window"])
    sampler_tr = DistributedSampler(
        ds_tr, num_replicas=dist_env.world_size, rank=dist_env.rank,
        shuffle=True, seed=int(seed))
    dl_tr = DataLoader(ds_tr, batch_size=lb, sampler=sampler_tr,
                       collate_fn=coll)
    sampler_va = _ShardSampler(len(ds_va), dist_env.world_size, dist_env.rank)
    dl_va = DataLoader(ds_va, batch_size=lb, sampler=sampler_va,
                       collate_fn=coll)

    # identical model init on every rank; per-rank dropout streams only AFTER
    # the DDP construction (the wrap itself must consume no rank-specific RNG)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    raw = ActSeqAttn(n_act=ds_tr.n_act, n_out=N_OUT, d=hpm["d_model"],
                     heads=hpm["heads"], depth=hpm["depth"], ffn=hpm["ffn"],
                     dropout=hpm["dropout"], pe=pe).to(dev)
    model = raw if dist_env.world_size == 1 else DistributedDataParallel(
        raw, device_ids=[dist_env.local_rank])
    torch.manual_seed(int(seed) + 10_000 * dist_env.rank)
    torch.cuda.manual_seed_all(int(seed) + 10_000 * dist_env.rank)

    opt = torch.optim.AdamW(raw.parameters(), lr=hpm["lr"], weight_decay=1e-5)
    warm = int(hpm["warmup"])
    best, best_state, bad = 1e9, None, 0
    for ep in range(n_ep):
        cur = (hpm["lr"] * (ep + 1) / warm if ep < warm else
               hpm["lr"] * 0.5 * (1 + math.cos(math.pi * (ep - warm)
                                               / max(n_ep - warm, 1))))
        for g in opt.param_groups:
            g["lr"] = cur
        sampler_tr.set_epoch(ep)
        model.train()
        for A, Y, m, P in dl_tr:
            A, Y = A.to(dev).float(), Y.to(dev).float()
            m, P = m.to(dev), P.to(dev).float()
            loss = global_masked_loss(model(A, P), Y, m, dist_env.world_size)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
        # global validation: local sums, then ONE detached all-reduce pair
        model.eval()
        num, den = 0.0, 0.0
        with torch.no_grad():
            for A, Y, m, P in dl_va:
                A, Y = A.to(dev).float(), Y.to(dev).float()
                m = m.to(dev)
                w = m.unsqueeze(-1).float()
                num += float((((model(A, P.to(dev).float()) - Y) ** 2)
                              * w).sum())
                den += float(w.sum())
        num_t = torch.tensor(num, device=dev)
        den_t = torch.tensor(den, device=dev)
        if dist_env.world_size > 1:
            torch.distributed.all_reduce(
                num_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                den_t, op=torch.distributed.ReduceOp.SUM)
        vl = float(num_t) / max(float(den_t), 1.0)
        # vl is identical on every rank -> the stop decision is synchronized
        if vl < best - 1e-7:
            best, best_state, bad = vl, copy.deepcopy(raw.state_dict()), 0
        else:
            bad += 1
            if bad >= int(hpm["patience"]):
                break
    if best_state is not None:
        raw.load_state_dict(best_state)
    n_params = int(sum(p.numel() for p in raw.parameters()))

    if dist_env.is_main:
        fp = run_fingerprint_payload
        if fp is None:
            fp = run_fingerprint(
                arm, int(seed), config_path, split_path, sidecar_dir, npz_dir,
                normalization_sha256=normalization_hash,
                source_audit_identity_sha256=(
                    source_audit_identity_sha256))
        validation_summary = {
            "arm": arm,
            "seed": int(seed),
            "epochs": n_ep,
            "stop_epoch": int(ep),
            "best_val_mse": float(best),
            "n_windows_train": len(ds_tr),
            "n_windows_val": len(ds_va),
            "n_params": n_params,
            "world_size": int(dist_env.world_size),
            "global_batch": int(hpm["batch"]),
            "run_fingerprint": fp,
        }
        artifact = {
            "state": raw.state_dict(), "n_act": ds_tr.n_act, "hp": hp,
            "pe": pe, "mean": mean, "std": std, "tgt_mean": tgt_mean,
            "tgt_std": tgt_std, "n_out": N_OUT,
            "normalization_sha256": normalization_hash,
            "best_val_mse": float(best), "seed": int(seed), "arm": arm,
            "run_fingerprint": fp, "n_params": n_params,
            "world_size": int(dist_env.world_size),
            "global_batch": int(hpm["batch"]),
            "validation_summary": validation_summary,
        }
        publish_training_artifacts(out_dir, artifact, validation_summary)
    dist_barrier(dist_env)
    return {"best_val_mse": float(best), "arm": arm, "seed": int(seed),
            "pe": pe, "n_act": ds_tr.n_act, "n_params": n_params,
            "n_windows_train": len(ds_tr), "n_windows_val": len(ds_va),
            "stop_epoch": int(ep), "epochs": n_ep,
            "world_size": int(dist_env.world_size),
            "global_batch": int(hpm["batch"])}
