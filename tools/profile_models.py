#!/usr/bin/env python
"""Profile parameters and convolution MACs for all three UNet variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_zoo.registry import MODEL_REGISTRY, create_model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model: nn.Module, input_shape: tuple[int, int, int, int]) -> int:
    """Count Conv2d MACs from real intermediate shapes on PyTorch's meta device."""
    total_macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
        nonlocal total_macs
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        per_output = (module.in_channels // module.groups) * kernel_height * kernel_width
        total_macs += int(batch * out_channels * out_height * out_width * per_output)

    meta_model = model.to("meta").eval()
    for layer in meta_model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
    try:
        with torch.inference_mode():
            meta_model(torch.empty(input_shape, device="meta"))
    finally:
        for handle in handles:
            handle.remove()
    return total_macs


def profile_model(name: str, input_shape: tuple[int, int, int, int]) -> dict[str, object]:
    model = create_model(name)
    return {
        "model": name,
        "params": count_parameters(model),
        "macs": count_macs(model, input_shape),
        "flops": 2 * count_macs(create_model(name), input_shape),
        "input_shape": "x".join(map(str, input_shape)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shape", nargs=4, type=int, default=(1, 4, 256, 256))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    input_shape = tuple(args.input_shape)
    if input_shape[1] != 4:
        parser.error("RAW input must have four channels")

    rows = [profile_model(name, input_shape) for name in MODEL_REGISTRY]
    for row in rows:
        print(
            f"{row['model']:<14} Params={row['params'] / 1e6:8.3f} M  "
            f"MACs={row['macs'] / 1e9:9.3f} G  FLOPs={row['flops'] / 1e9:9.3f} G"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
