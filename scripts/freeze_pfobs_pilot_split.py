# -*- coding: utf-8 -*-
"""Freeze the pilot-only random by-shot split manifest (607/76/76).

Reproduces the split the training pipeline already pins: every existing
caller of ``bench.load_filtered_split`` relies on its defaults (val 0.1,
test 0.1, split seed 0) on the 759 NpzGeom shots, so this manifest labels
the membership already in use instead of inventing a new one. The JSON
carries ``claim_scope: pilot_only``: it may back unit tests, smoke runs,
resource estimation and the pilot only. It can never support the
submission's prospective generalization claim -- that needs the frozen
campaign manifest (``configs/splits/communications_physics_campaign_v1.json``,
supplied by Essential Work 1), which this script never creates.

Refuses to overwrite an existing manifest whose content differs unless
``--replace`` is passed; rewriting identical content is a no-op.

Usage:
  python scripts/freeze_pfobs_pilot_split.py [--replace]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml import bench  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT_MANIFEST = ROOT / "configs" / "splits" / "pfobs_random_pilot.json"
# Supplied and frozen by Essential Work 1; deliberately never written here.
FINAL_CAMPAIGN_MANIFEST = ROOT / "configs" / "splits" / "communications_physics_campaign_v1.json"
# The pinned split membership the newtrain pipeline already scores against:
# load_filtered_split's defaults (val 0.1 / test 0.1 / seed 0) on 759 shots.
EXPECTED_COUNTS = (607, 76, 76)


def build_payload(npz_dir):
    """The pilot manifest payload for one NpzGeom-style dataset dir.

    Membership comes from ``bench.load_filtered_split`` with its default
    split seed 0 -- the same pinned seed every training caller already
    uses -- so the frozen arrays are the membership in use, not a new
    draw.
    """
    train, validation, test = bench.load_filtered_split(npz_dir)
    return {
        "name": "pfobs_random_pilot",
        "version": 1,
        "time_axis": "native_gmag_bnd",
        "claim_scope": "pilot_only",
        "train": [int(s) for s in train],
        "validation": [int(s) for s in validation],
        "test": [int(s) for s in test],
        "shot_metadata": None,
        "slice_strata_dir": None,
    }


def write_manifest(out_path, payload, replace=False):
    """Freeze ``payload`` at ``out_path`` (stable serialization).

    An existing file with identical content is left byte-for-byte alone;
    a differing file is refused unless ``replace``. Returns ``"wrote"``,
    ``"unchanged"`` or ``"replaced"``.
    """
    out_path = pathlib.Path(out_path)
    text = json.dumps(payload, indent=2)
    if out_path.exists():
        if json.loads(out_path.read_text()) == payload:
            print(f"{out_path}: already frozen with identical content")
            return "unchanged"
        if not replace:
            raise SystemExit(
                f"{out_path} exists and differs from the newly generated "
                f"manifest; pass --replace to refreeze it deliberately")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return "replaced" if replace else "wrote"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replace", action="store_true",
                    help="refreeze a differing existing manifest")
    args = ap.parse_args()

    npz_dir = get_proj_config().npzgeom_dir
    payload = build_payload(npz_dir)
    counts = tuple(len(payload[k]) for k in ("train", "validation", "test"))
    if counts != EXPECTED_COUNTS:
        raise SystemExit(
            f"refusing to freeze: {npz_dir} gives train/validation/test = "
            f"{counts}, expected {EXPECTED_COUNTS} -- the dataset changed "
            f"under us; a new split needs a new manifest name, not a "
            f"silent refreeze of this one")
    status = write_manifest(PILOT_MANIFEST, payload, replace=args.replace)
    print(f"{PILOT_MANIFEST}: {status} "
          f"(claim_scope={payload['claim_scope']}, {counts[0]}/"
          f"{counts[1]}/{counts[2]} train/validation/test, split seed 0)")


if __name__ == "__main__":
    main()
