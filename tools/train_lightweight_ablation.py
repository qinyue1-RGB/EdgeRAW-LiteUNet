#!/usr/bin/env python
"""Validate and launch the three fair lightweight-ablation training runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "configs/lightweight/original_unet.yaml",
    ROOT / "configs/lightweight/liteunet_v1.yaml",
    ROOT / "configs/lightweight/liteunet_v2.yaml",
]
ALLOWED_DIFFERENCES = {"network", "log_dir", "checkpoint_dir", "tensorboard_dir", "valid_dir"}


def validate_configs() -> None:
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in CONFIGS]
    reference = {key: value for key, value in configs[0].items() if key not in ALLOWED_DIFFERENCES}
    for path, config in zip(CONFIGS[1:], configs[1:]):
        comparable = {key: value for key, value in config.items() if key not in ALLOWED_DIFFERENCES}
        if comparable != reference:
            raise ValueError(f"Fairness check failed: hyperparameters differ in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_configs()

    index_path = ROOT / "IMX766/train_data/train_data_idx.txt"
    if not index_path.exists() and not args.dry_run:
        raise FileNotFoundError(
            "IMX766 training data is absent. Download it as documented in IMX766/data_here.txt first."
        )
    for config in CONFIGS:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            str(ROOT / "train_model/train.py"),
            "--config",
            str(config),
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
