# -*- coding: utf-8 -*-
"""The capacity-matched m3-small config.

The whole point of this arm is a parameter count in the same order as m2-GRU's
53,730. If the config drifts, the "capacity-matched" claim silently becomes false,
so the count is pinned to an exact number here rather than described in a comment.

Spec: docs/superpowers/specs/2026-08-07-capacity-matched-multiseed-design.md 3.1, 7
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dcs_features import load_dcs_config, strict_channels  # noqa: E402
from src.ml.models import ActSeqAttn, ActSeqGRU  # noqa: E402
from src.ml.target import N_OUT  # noqa: E402

CFG = "configs/dcs_model_attn_small.yml"
HP_KEYS = {"pe", "d_model", "heads", "depth", "ffn", "dropout", "window", "ctx",
           "batch", "lr", "warmup", "epochs", "patience"}


def test_config_loads_with_every_key_the_trainer_reads():
    c = load_dcs_config(CFG)
    assert HP_KEYS - set(c["hp"]["m3"]) == set()
    assert c["hp"]["m3"]["window"] - c["hp"]["m3"]["ctx"] == 1536


def test_eighteen_strict_actuators_and_no_positional_columns():
    """Positional info must enter through the architecture, never as a column."""
    ch = [n for n, _k, _p in strict_channels(load_dcs_config(CFG))]
    assert len(ch) == 18
    assert not [n for n in ch if n.startswith("pe_") or n == "dt_gap_s"]


def test_small_capacity_is_pinned_and_close_to_the_gru():
    """103,778 params vs the GRU's 53,730 = 1.93x -- the capacity-match claim."""
    hp = load_dcs_config(CFG)["hp"]["m3"]
    small = ActSeqAttn(n_act=22, n_out=N_OUT, d=hp["d_model"], heads=hp["heads"],
                       depth=hp["depth"], ffn=hp["ffn"], dropout=hp["dropout"],
                       pe=hp["pe"])
    n_small = sum(p.numel() for p in small.parameters())
    gru = ActSeqGRU(n_act=22, n_out=N_OUT, hidden=64, layers=2)
    n_gru = sum(p.numel() for p in gru.parameters())
    assert n_small == 103_778, f"capacity drifted: {n_small:,}"
    assert n_gru == 53_730, f"the GRU baseline changed: {n_gru:,}"
    assert n_small / n_gru < 2.5, "no longer a capacity match"


def test_only_capacity_differs_from_the_big_config():
    """Everything except d_model/heads/depth/ffn must match dcs_model_attn.yml."""
    small = load_dcs_config(CFG)["hp"]["m3"]
    big = load_dcs_config("configs/dcs_model_attn.yml")["hp"]["m3"]
    for k in ("pe", "dropout", "window", "ctx", "batch", "lr", "warmup",
              "epochs", "patience"):
        assert small[k] == big[k], f"{k} differs: {small[k]} vs {big[k]}"
    assert (small["d_model"], small["heads"], small["depth"], small["ffn"]) \
        == (64, 4, 2, 256)
