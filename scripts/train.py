# -*- coding: utf-8 -*-
"""torchrun entry for training. Dispatches to the baseline or sequence trainer.

    torchrun --nproc_per_node=4 scripts/train.py baseline --epochs 50 --max_shots 100
    torchrun --nproc_per_node=4 scripts/train.py seq --family tcn --W 64 --epochs 50

The subcommand is stripped from argv before the trainer parses its own flags.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.train import main, seq_main  # noqa: E402

MODES = {"baseline": main, "seq": seq_main}

if __name__ == "__main__":
    mode = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if mode not in MODES:
        sys.exit(f"usage: train.py {{{'|'.join(MODES)}}} [args...]")
    MODES[mode]()
