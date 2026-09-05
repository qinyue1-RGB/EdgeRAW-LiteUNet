#!/usr/bin/env python
"""Create honest, GitHub-ready plots from lightweight_ablation.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = ["#0072B2", "#E69F00", "#009E73"]
HATCHES = ["//", "\\\\", "xx"]


def read_rows(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open(encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            parsed: dict[str, object] = dict(raw)
            for key in ("params", "macs", "psnr", "ssim", "latency_ms", "fps"):
                value = raw.get(key, "")
                parsed[key] = float(value) if value not in ("", "N/A", "nan") else math.nan
            rows.append(parsed)
    return rows


def save_bar(rows, key: str, ylabel: str, output: Path, scale: float = 1.0) -> None:
    names = [str(row["model"]) for row in rows]
    values = [float(row[key]) / scale for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    if all(math.isnan(value) for value in values):
        ax.text(0.5, 0.55, "Not available", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.43, "Validation data and trained checkpoints are required", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        bars = ax.bar(names, values, color=PALETTE, edgecolor="#222222", linewidth=0.8)
        for bar, hatch, value in zip(bars, HATCHES, values):
            bar.set_hatch(hatch)
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def save_tradeoff(rows, output: Path) -> None:
    points = [(str(row["model"]), float(row["latency_ms"]), float(row["psnr"])) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    available = [(name, x, y) for name, x, y in points if not math.isnan(x) and not math.isnan(y)]
    if not available:
        ax.text(0.5, 0.55, "Not available", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.43, "PSNR requires validation data and trained checkpoints", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        max_latency = max(latency for _, latency, _ in available)
        max_psnr = max(psnr for _, _, psnr in available)
        for index, (name, latency, psnr) in enumerate(available):
            ax.scatter(latency, psnr, s=85, color=PALETTE[index], marker=["o", "s", "^"][index], label=name)
            is_rightmost = latency == max_latency
            is_topmost = psnr == max_psnr
            ax.annotate(
                name,
                (latency, psnr),
                xytext=(-6 if is_rightmost else 6, -8 if is_topmost else 6),
                ha="right" if is_rightmost else "left",
                va="top" if is_topmost else "bottom",
                textcoords="offset points",
            )
        ax.set(xlabel="Latency (ms / image)", ylabel="PSNR (dB)")
        ax.grid(alpha=0.25)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def generate_plots(csv_path: Path, output_dir: Path) -> None:
    rows = read_rows(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_bar(rows, "params", "Parameters (million)", output_dir / "params.png", 1e6)
    save_bar(rows, "macs", "MACs (billion)", output_dir / "macs.png", 1e9)
    save_bar(rows, "psnr", "PSNR (dB)", output_dir / "psnr.png")
    save_bar(rows, "latency_ms", "Latency (ms / image)", output_dir / "latency.png")
    save_tradeoff(rows, output_dir / "psnr_vs_latency.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("artifacts/lightweight_ablation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/lightweight"))
    args = parser.parse_args()
    generate_plots(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
