# -*- coding: utf-8 -*-
"""DDP training of one PF-context ``(context, seed)`` run.

The pf-context analogue of :mod:`src.ml.pfobs_train`. ``train_one`` is called
by the matrix runner (Task 5), which owns the process group: one
``torchrun`` job initializes it once and then calls ``train_one`` once per
``(context, seed)``; ``train_one`` never initializes or destroys the group.

The effective global batch is exactly 16 for every run:
``world_size * microbatch * gradient_accumulation == 16``. DDP averages
gradients, so each local masked SSE is scaled by ``world_size /
global_denominator`` (the denominator reduced across ranks and the whole
accumulation group BEFORE any backward call); after averaging, the step
minimizes the masked MSE of the entire global batch. ``no_sync`` holds every
microbatch but the last of a group, and ``zero_grad`` precedes each group.

Model init is seeded identically on every rank; dropout streams are reseeded
per rank (``seed + 10_000 * rank``) inside ``run_epochs``, i.e. only after
the DDP construction, exactly as the upstream trainer does.

Provenance channel: ``build_context_loaders`` is the only helper that sees
the data directories, so it attaches the target/sidecar meta hashes to the
train loader; ``run_epochs`` moves them plus the scored-index hash into its
result dict, which the verbatim ``train_one`` hands to the rank-0 artifact
writer together with everything else the artifact records.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .models import ActSeqAttn
from .pf_context import (
    CONTEXT_LEVELS, existing_source_files, layer_attention_mask,
)
from .pf_observability import sha256_file, sha256_tree
from .pfctx_data import PFContextDataset, pad_context_collate
from .pfobs_train import DistEnv, init_dist  # noqa: F401  (re-exported)

INPUT_WIDTH = 21          # the fixed Arm B block: [zero(10), actual(10), Ip]
N_OUT = 34                # 32 radii + absolute (Rgeom, Zgeom)

# This worktree's root (pfctx_train.py is a real file here; the upstream
# pf_observability symlink resolves elsewhere, so never reuse its root).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# The artifact writer hashes exactly the entries of the AUTHORITATIVE frozen
# list (PFCTX_SOURCE_FILES in pf_context.py) that exist when this module is
# imported: the Task-6 inference module and scorer script join the digest
# automatically once created, and the frozen list itself never changes.
SOURCE_FILES = existing_source_files(PROJECT_ROOT)


def microbatch_contract(effective_global_batch, world_size, microbatch):
    denom = int(world_size) * int(microbatch)
    if denom <= 0 or int(effective_global_batch) % denom:
        raise ValueError(
            "effective global batch must be divisible by world_size * microbatch")
    accumulation = int(effective_global_batch) // denom
    return accumulation, denom * accumulation


def validate_context_config(cfg):
    required = {
        "study": "pf_context", "arm": "B", "model": "ActSeqAttn",
        "pe": "rope_time", "time_axis": "native_gmag_bnd",
        "input_width": 21, "score_block": 512,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"{key} must equal {expected!r}")
    fixed_hp = {
        "d_model": 256, "heads": 8, "depth": 6, "ffn": 1024,
        "dropout": 0.1, "effective_global_batch": 16,
        "lr": 0.0003, "warmup": 5, "epochs": 80,
        "patience": 12, "gradient_clip": 1.0,
    }
    for key, expected in fixed_hp.items():
        if cfg["hp"].get(key) != expected:
            raise ValueError(f"hp.{key} must equal {expected!r}")
    fixed_selection = {
        "anchor": "h2048", "practical_margin_mm": 1.0,
        "validation_family_alpha": 0.05,
        "bootstrap_resamples": 10000,
        "required_seed_agreement": 4,
    }
    for key, expected in fixed_selection.items():
        if cfg["selection"].get(key) != expected:
            raise ValueError(f"selection.{key} must equal {expected!r}")
    got = [(x["label"], int(x["nominal_samples"]), float(x["seconds"]))
           for x in cfg["contexts"]]
    expected = [(x.label, x.nominal_samples, x.seconds)
                for x in CONTEXT_LEVELS]
    if got != expected:
        raise ValueError("context grid differs from the frozen specification")
    if list(map(int, cfg["seeds"])) != [0, 1, 2, 3, 4]:
        raise ValueError("seeds must be exactly 0,1,2,3,4")
    accumulation, effective = microbatch_contract(
        cfg["hp"]["effective_global_batch"], 4,
        cfg["hp"]["microbatch_per_rank"])
    if (cfg["hp"]["production_gradient_accumulation"] != accumulation or
            effective != 16):
        raise ValueError(
            "production microbatch and accumulation must preserve global batch 16")


def masked_numerator(pred, target, loss_mask):
    weight = loss_mask.unsqueeze(-1).to(pred.dtype)
    numerator = (((pred - target) ** 2) * weight).sum()
    denominator = weight.sum()
    return numerator, denominator


# ── distributed wrapping and the two samplers ────────────────────────
class _WorldOne:
    """The world-1 stand-in for DDP.

    The pfctx loop accumulates microbatches even at world 1 (e.g. the CPU
    smoke: microbatch 4 on one rank), and the exact-loss backward calls
    ``ddp_model.no_sync()`` for every microbatch but the last -- a raw
    ``ActSeqAttn`` has no such method. This wrapper provides the DDP call
    surface (``module``, ``no_sync``, and delegation for everything else)
    with zero collectives; ``wrap_ddp`` returns it only at world 1, where DDP
    itself would need a process group that does not exist.
    """

    def __init__(self, module):
        self.module = module

    def no_sync(self):
        return contextlib.nullcontext()

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.module, name)


def wrap_ddp(model, dist_env):
    """Wrap for DDP: identical init on every rank, NCCL device ids on CUDA,
    plain Gloo-compatible DDP on CPU, ``_WorldOne`` at world 1."""
    if dist_env.world_size == 1:
        return _WorldOne(model)
    if dist_env.device.type == "cuda":
        return DistributedDataParallel(
            model, device_ids=[dist_env.local_rank])
    return DistributedDataParallel(model)


class _ShardSampler(Sampler):
    """Rank ``r`` of ``w`` sees ``[r::w]`` -- disjoint, with NO padding.

    Mirrors the upstream trainer's validation sampler: a padded validation
    window would enter the globally reduced numerator/denominator twice, so
    validation shards exactly once while training pads (``DistributedSampler``
    with ``drop_last=False`` repeats indices deterministically, keeping every
    rank's batch count equal -- which is what makes the last partial
    accumulation group safe).
    """

    def __init__(self, n, world_size, rank):
        self.n, self.world_size, self.rank = int(n), int(world_size), int(rank)

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world_size))

    def __len__(self):
        return max(0, (self.n - self.rank + self.world_size - 1)
                   // self.world_size)


def build_context_loaders(target_dir, sidecar_dir, train_shots,
                          validation_shots, context, feature_stats,
                          target_stats, microbatch, seed, dist_env):
    """The train/validation loaders for one ``(context, seed)`` fit.

    Train: ``DistributedSampler(drop_last=False)`` -- the per-rank window
    index is padded deterministically to a multiple of the world size so
    every rank runs the same number of batches. Validation: the exact-once
    ``_ShardSampler``. Both collate with ``pad_context_collate`` at the
    per-rank microbatch size.

    The train loader carries ``pfctx_meta_fingerprint`` (the target/sidecar
    ``meta.json`` hashes); ``run_epochs`` moves it into its result so the
    rank-0 artifact writer can record where the data came from.
    """
    feature_mean, feature_std = feature_stats
    target_mean, target_std = target_stats
    train_dataset = PFContextDataset(
        target_dir, sidecar_dir, train_shots, context,
        feature_mean, feature_std, target_mean, target_std)
    validation_dataset = PFContextDataset(
        target_dir, sidecar_dir, validation_shots, context,
        feature_mean, feature_std, target_mean, target_std)
    microbatch = int(microbatch)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist_env.world_size, rank=dist_env.rank,
        shuffle=True, seed=int(seed), drop_last=False)
    train_loader = DataLoader(
        train_dataset, batch_size=microbatch, sampler=train_sampler,
        collate_fn=pad_context_collate)
    validation_loader = DataLoader(
        validation_dataset, batch_size=microbatch,
        sampler=_ShardSampler(len(validation_dataset), dist_env.world_size,
                              dist_env.rank),
        collate_fn=pad_context_collate)
    train_loader.pfctx_meta_fingerprint = {
        "target_meta_sha256": sha256_file(
            pathlib.Path(target_dir) / "meta.json"),
        "sidecar_meta_sha256": sha256_file(
            pathlib.Path(sidecar_dir) / "meta.json"),
    }
    return train_loader, validation_loader


def backward_accumulation_group(ddp_model, batches, optimizer, dist_env,
                                context_seconds, depth, gradient_clip):
    local_den = torch.zeros((), device=dist_env.device)
    for batch in batches:
        local_den += batch.loss_mask.sum().to(dist_env.device)
    global_den = local_den.clone()
    if dist_env.world_size > 1:
        torch.distributed.all_reduce(global_den, op=torch.distributed.ReduceOp.SUM)
    if global_den.item() <= 0:
        raise RuntimeError("accumulation group contains no valid target rows")

    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(batches):
        sync = index == len(batches) - 1
        guard = contextlib.nullcontext() if sync else ddp_model.no_sync()
        with guard:
            features = batch.features.to(dist_env.device)
            target = batch.target.to(dist_env.device)
            loss_mask = batch.loss_mask.to(dist_env.device)
            time = batch.time.to(dist_env.device)
            history = batch.history_valid.to(dist_env.device)
            real = batch.real.to(dist_env.device)
            position = batch.position.to(dist_env.device)
            attention = layer_attention_mask(
                time, history, real, context_seconds, depth)
            pred = ddp_model(features, position, attn_mask=attention)
            numerator, _denominator = masked_numerator(
                pred, target, loss_mask)
            loss = numerator * dist_env.world_size / global_den
            loss.backward()
    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), gradient_clip)
    optimizer.step()


# ── the epoch loop and the engineering readouts ──────────────────────
def _accumulation_groups(loader, accumulation):
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) >= int(accumulation):
            yield group
            group = []
    if group:
        yield group


def _assert_equal_batch_groups(local_groups, dist_env):
    """The last partial accumulation group is only safe when every rank runs
    the same number of optimizer groups. ``DistributedSampler`` guarantees
    it; this asserts it across ranks so a foreign sampler fails loudly
    instead of deadlocking the first backward."""
    if dist_env.world_size == 1:
        return
    lo = torch.tensor(int(local_groups), dtype=torch.int64,
                      device=dist_env.device)
    hi = lo.clone()
    torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX)
    if int(lo.item()) != int(hi.item()):
        raise RuntimeError(
            f"rank {dist_env.rank}: optimizer-group counts differ across "
            f"ranks (min {int(lo.item())}, max {int(hi.item())}) -- the "
            "last partial accumulation group would deadlock DDP")


def run_epochs(ddp_model, optimizer, train_loader, validation_loader,
               context, seed, hp, accumulation, dist_env):
    """Warm-up + cosine epochs over exact accumulated global batches.

    Per epoch: the frozen schedule (linear warm-up then half-cosine decay),
    ``set_epoch`` on the distributed train sampler, one
    ``backward_accumulation_group`` per group of consecutive microbatches
    (the trailing partial group included -- safe because every rank's batch
    count is asserted equal), then ONE globally reduced validation
    numerator/denominator pair so the early-stop decision is identical on
    every rank. Returns the metrics plus the dataset-derived provenance
    (scored-index and meta hashes) the artifact writer serializes.
    """
    raw = (ddp_model.module if hasattr(ddp_model, "module") else ddp_model)
    n_epochs, warmup, patience = (
        int(hp["epochs"]), int(hp["warmup"]), int(hp["patience"]))
    lr, depth, clip = (
        float(hp["lr"]), int(hp["depth"]), float(hp["gradient_clip"]))
    _assert_equal_batch_groups(
        math.ceil(len(train_loader) / max(int(accumulation), 1)), dist_env)
    # per-rank dropout streams, reseeded only AFTER the DDP construction
    torch.manual_seed(int(seed) + 10_000 * dist_env.rank)
    torch.cuda.manual_seed_all(int(seed) + 10_000 * dist_env.rank)

    best, best_state, best_epoch, bad, stop_epoch = 1e9, None, -1, 0, n_epochs - 1
    for epoch in range(n_epochs):
        current = (lr * (epoch + 1) / warmup if epoch < warmup else
                   lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup)
                                           / max(n_epochs - warmup, 1))))
        for group in optimizer.param_groups:
            group["lr"] = current
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        for group in _accumulation_groups(train_loader, accumulation):
            backward_accumulation_group(
                ddp_model, group, optimizer, dist_env, context.seconds,
                depth, clip)
        # global validation: local sums, then ONE detached all-reduce pair
        ddp_model.eval()
        numerator_sum, denominator_sum = 0.0, 0.0
        with torch.no_grad():
            for batch in validation_loader:
                attention = layer_attention_mask(
                    batch.time.to(dist_env.device),
                    batch.history_valid.to(dist_env.device),
                    batch.real.to(dist_env.device), context.seconds, depth)
                pred = ddp_model(
                    batch.features.to(dist_env.device),
                    batch.position.to(dist_env.device), attn_mask=attention)
                numerator, denominator = masked_numerator(
                    pred, batch.target.to(dist_env.device),
                    batch.loss_mask.to(dist_env.device))
                numerator_sum += float(numerator)
                denominator_sum += float(denominator)
        num_t = torch.tensor(numerator_sum, device=dist_env.device)
        den_t = torch.tensor(denominator_sum, device=dist_env.device)
        if dist_env.world_size > 1:
            torch.distributed.all_reduce(
                num_t, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                den_t, op=torch.distributed.ReduceOp.SUM)
        val_loss = float(num_t) / max(float(den_t), 1.0)
        # val_loss is identical on every rank -> the stop decision is synced
        if val_loss < best - 1e-7:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = copy.deepcopy(raw.state_dict())
        else:
            bad += 1
            if bad >= patience:
                stop_epoch = epoch
                break
    if best_state is not None:
        raw.load_state_dict(best_state)
    return {
        "best_val_mse": float(best),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epoch) + 1,
        "stop_epoch": int(stop_epoch),
        "n_train_windows": len(train_loader.dataset),
        "n_validation_windows": len(validation_loader.dataset),
        "scored_index_sha256": scored_index_sha256(
            train_loader.dataset, validation_loader.dataset),
        "target_meta_sha256": train_loader.pfctx_meta_fingerprint[
            "target_meta_sha256"],
        "sidecar_meta_sha256": train_loader.pfctx_meta_fingerprint[
            "sidecar_meta_sha256"],
    }


def mean_sequence_tokens(dataset):
    """Mean window length (real tokens per item) over the dataset's index."""
    if len(dataset.index) == 0:
        return 0.0
    total = sum(int(window.window_end - window.window_start)
                for _series, window in dataset.index)
    return total / len(dataset.index)


def profile_attention_backend(ddp_model, validation_loader, context,
                              dist_env):
    """Sorted attention operator names from exactly ONE no-grad batch.

    Engineering readout only (spec 6.4 / 9.2): which attention kernels the
    masked elapsed-time path selected. It never touches the training path --
    eval mode, no gradient, one validation batch, no optimizer -- and
    returns an empty list on CPU backends that report no matching operator.

    A rank whose OWN ``_ShardSampler`` shard is empty skips the probe: the
    no-grad eval forward performs no collectives (DDP all-reduces only in
    backward, and buffer sync requires a grad-enabled forward), so the
    rank-local skip leaves nobody waiting -- exactly how the validation
    loop iterates its own possibly-empty shard before the one all-reduce.
    """
    if len(validation_loader.dataset) == 0:
        return []
    if len(validation_loader) == 0:
        return []          # this rank's shard yields no batch to profile
    raw = (ddp_model.module if hasattr(ddp_model, "module") else ddp_model)
    depth = len(raw.attn)
    ddp_model.eval()
    batch = next(iter(validation_loader))
    attention = layer_attention_mask(
        batch.time.to(dist_env.device),
        batch.history_valid.to(dist_env.device),
        batch.real.to(dist_env.device), context.seconds, depth)
    activities = ([torch.profiler.ProfilerActivity.CUDA]
                  if dist_env.device.type == "cuda"
                  else [torch.profiler.ProfilerActivity.CPU])
    with torch.profiler.profile(activities=activities) as profiler:
        with torch.no_grad():
            ddp_model(batch.features.to(dist_env.device),
                      batch.position.to(dist_env.device), attn_mask=attention)
    return sorted({event.key for event in profiler.key_averages()
                   if "scaled_dot_product" in event.key.lower()
                   or "attention" in event.key.lower()})


# ── the content fingerprints the artifact records ────────────────────
def _canonical_sha256(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def config_content_sha256(config):
    """sha256 over the canonical JSON of the parsed config dict (the trainer
    receives the dict, not the file; the runner's file-level hash is the
    Task-5 fingerprint's job)."""
    return _canonical_sha256(config)


def split_content_sha256(split):
    payload = {
        "name": str(split.name), "version": int(split.version),
        "time_axis": str(split.time_axis),
        "train": [int(s) for s in split.train],
        "validation": [int(s) for s in split.validation],
        "test": [int(s) for s in split.test],
        "shot_metadata": (None if split.shot_metadata is None
                          else str(split.shot_metadata)),
        "slice_strata_dir": (None if split.slice_strata_dir is None
                             else str(split.slice_strata_dir)),
    }
    return _canonical_sha256(payload)


def normalization_sha256(feature_stats, target_stats):
    digest = hashlib.sha256()
    for array in (*feature_stats, *target_stats):
        data = np.ascontiguousarray(array)
        digest.update(f"{data.dtype.str}:{data.shape};".encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def scored_index_sha256(train_dataset, validation_dataset):
    """sha256 over the exact (role, shot, window, block) tiling both datasets
    expose -- the optimizer's and the validator's scored-row index, which a
    context level must never silently change."""
    lines = []
    for role, dataset in (("train", train_dataset),
                          ("validation", validation_dataset)):
        for series_index, window in dataset.index:
            shot = dataset.series[series_index][0]
            lines.append(f"{role}:{int(shot)}:{window.window_start}:"
                         f"{window.window_end}:{window.block_start}:"
                         f"{window.block_end}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ── the rank-0 artifact ──────────────────────────────────────────────
REQUIRED_ARTIFACT_KEYS = (
    "study", "arm", "model", "pe", "time_axis", "n_act", "n_out",
    "context_label", "nominal_samples", "context_seconds",
    "per_layer_seconds", "score_block", "seed", "world_size",
    "effective_global_batch", "microbatch", "gradient_accumulation",
    "depth", "claim_scope", "state", "feature_mean", "feature_std",
    "target_mean", "target_std", "best_val_mse", "best_epoch",
    "stop_epoch", "epochs_completed", "n_train_windows",
    "n_validation_windows", "elapsed_seconds", "seconds_per_epoch",
    "mean_sequence_tokens", "peak_memory_bytes", "attention_backend_ops",
    "fingerprints",
)


def _validate_artifact_payload(artifact, config):
    """Reload validation: every required key present, and the stored state
    reconstructs the fixed production model strictly."""
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"artifact is missing {missing}")
    hp = config["hp"]
    probe = ActSeqAttn(
        n_act=INPUT_WIDTH, n_out=N_OUT, d=hp["d_model"],
        heads=hp["heads"], depth=hp["depth"], ffn=hp["ffn"],
        dropout=hp["dropout"], pe="rope_time")
    probe.load_state_dict(artifact["state"])
    return artifact


def write_context_artifact_atomic(output_dir, model, result, context, seed,
                                  split, config, feature_stats, target_stats,
                                  world_size, effective, accumulation):
    """Rank 0 writes ``<output_dir>/m3.pt`` atomically.

    ``output_dir`` IS the run directory (the runner passes
    ``ProjDB/trains/<run_name>``); this writer creates it. Serialization goes
    to a sibling ``m3.pt.building``; the rename happens only after the file
    is reloaded and validated (keys + state reconstruction), so a partially
    written or corrupt artifact is never published. A failed validation
    leaves the ``.building`` file for diagnosis.
    """
    hp = config["hp"]
    artifact = {
        "study": "pf_context", "arm": "B", "model": "ActSeqAttn",
        "pe": "rope_time", "time_axis": "native_gmag_bnd",
        "n_act": INPUT_WIDTH, "n_out": N_OUT,
        "context_label": str(context.label),
        "nominal_samples": int(context.nominal_samples),
        "context_seconds": float(context.seconds),
        "per_layer_seconds": float(context.per_layer_seconds),
        "score_block": int(config["score_block"]),
        "seed": int(seed), "world_size": int(world_size),
        "effective_global_batch": int(effective),
        "microbatch": int(hp["microbatch_per_rank"]),
        "gradient_accumulation": int(accumulation),
        "depth": int(hp["depth"]),
        "claim_scope": getattr(split, "claim_scope", None),
        "state": model.state_dict(),
        "feature_mean": np.asarray(feature_stats[0], np.float32),
        "feature_std": np.asarray(feature_stats[1], np.float32),
        "target_mean": np.asarray(target_stats[0], np.float32),
        "target_std": np.asarray(target_stats[1], np.float32),
        "best_val_mse": float(result["best_val_mse"]),
        "best_epoch": int(result["best_epoch"]),
        "stop_epoch": int(result["stop_epoch"]),
        "epochs_completed": int(result["epochs_completed"]),
        "n_train_windows": int(result["n_train_windows"]),
        "n_validation_windows": int(result["n_validation_windows"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "seconds_per_epoch": float(result["seconds_per_epoch"]),
        "mean_sequence_tokens": float(result["mean_sequence_tokens"]),
        "peak_memory_bytes": int(result["peak_memory_bytes"]),
        "attention_backend_ops": list(result["attention_backend_ops"]),
        "fingerprints": {
            "config_sha256": config_content_sha256(config),
            "split_sha256": split_content_sha256(split),
            "source_sha256": sha256_tree(PROJECT_ROOT, include=SOURCE_FILES),
            "sidecar_meta_sha256": str(result["sidecar_meta_sha256"]),
            "target_meta_sha256": str(result["target_meta_sha256"]),
            "normalization_sha256": normalization_sha256(
                feature_stats, target_stats),
            "scored_index_sha256": str(result["scored_index_sha256"]),
        },
    }
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "m3.pt"
    building = path.with_name(path.name + ".building")
    torch.save(artifact, building)
    reloaded = torch.load(building, map_location="cpu", weights_only=False)
    _validate_artifact_payload(reloaded, config)
    os.replace(building, path)
    return path


def train_one(context, seed, split, config, target_dir, sidecar_dir,
              output_dir, dist_env, feature_stats, target_stats):
    validate_context_config(config)
    if context.label not in {x.label for x in CONTEXT_LEVELS}:
        raise ValueError(f"unfrozen context {context.label!r}")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    hp = config["hp"]
    accumulation, effective = microbatch_contract(
        hp["effective_global_batch"], dist_env.world_size,
        hp["microbatch_per_rank"])
    if (dist_env.world_size == 4 and
            int(hp["production_gradient_accumulation"]) != accumulation):
        raise ValueError(
            "production accumulation does not preserve global batch 16")
    train_loader, validation_loader = build_context_loaders(
        target_dir, sidecar_dir, split.train, split.validation,
        context, feature_stats, target_stats,
        microbatch=hp["microbatch_per_rank"], seed=int(seed),
        dist_env=dist_env)
    model = ActSeqAttn(
        n_act=21, n_out=34, d=hp["d_model"], heads=hp["heads"],
        depth=hp["depth"], ffn=hp["ffn"],
        dropout=hp["dropout"], pe="rope_time").to(dist_env.device)
    ddp_model = wrap_ddp(model, dist_env)
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=hp["lr"], weight_decay=1e-5)
    if dist_env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dist_env.device)
    started = time.perf_counter()
    result = run_epochs(
        ddp_model, optimizer, train_loader, validation_loader,
        context, seed, hp, accumulation, dist_env)
    elapsed = time.perf_counter() - started
    result["elapsed_seconds"] = elapsed
    result["seconds_per_epoch"] = elapsed / max(result["epochs_completed"], 1)
    result["mean_sequence_tokens"] = mean_sequence_tokens(train_loader.dataset)
    result["peak_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(dist_env.device))
        if dist_env.device.type == "cuda" else 0)
    result["attention_backend_ops"] = profile_attention_backend(
        ddp_model, validation_loader, context, dist_env)
    if dist_env.is_main:
        write_context_artifact_atomic(
            output_dir, model, result, context, seed, split, config,
            feature_stats, target_stats, dist_env.world_size,
            effective, accumulation)
    return result
