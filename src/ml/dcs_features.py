# -*- coding: utf-8 -*-
"""DCS strict-actuator feature layer for MergedNpz.

Reads the 18 strict-actuator columns from each NPZ (heating NaN->0, essential
NaN->drop), adds engineered snapshot features, and yields the actuator series
for the temporal model. Config-driven via configs/dcs_model.yml.
"""
import json
import pathlib

import numpy as np
import yaml

from ..proj_config import get_proj_config


def load_dcs_config():
    cfg = get_proj_config()
    with open(cfg.base_dir / "configs" / "dcs_model.yml") as f:
        return yaml.safe_load(f)


def load_meta(npz_dir):
    return json.loads((pathlib.Path(npz_dir) / "meta.json").read_text())


def node_col_map(meta):
    """{node_name: X column index} from the meta inputs layout."""
    out = {}
    for g in meta["inputs"]:
        for i, node in enumerate(g["nodes"]):
            out[node] = int(g["cols"][0]) + i
    return out


def strict_channels(cfg):
    """[(node, kind, nan_policy)] in config order."""
    return [(c["node"], c["kind"], c["nan"]) for c in cfg["channels"]]
