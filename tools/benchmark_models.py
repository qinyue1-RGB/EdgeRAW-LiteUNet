#!/usr/bin/env python
"""Unified architecture, quality and runtime benchmark for the UNet ablation."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_zoo.registry import MODEL_REGISTRY, create_model
from tools.plot_lightweight_results import generate_plots
from tools.profile_models import profile_model

CHECKPOINT_PATHS = {
    "Original UNet": Path("artifacts/training/original_unet/checkpoints/Original UNet_best_ckpt.pth"),
    "LiteUNet-v1": Path("artifacts/training/liteunet_v1/checkpoints/LiteUNet-v1_best_ckpt.pth"),
    "LiteUNet-v2": Path("artifacts/training/liteunet_v2/checkpoints/LiteUNet-v2_best_ckpt.pth"),
}


def load_state_dict(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    model = model.to(device).eval()
    inputs = torch.randn(input_shape, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(inputs)
            end.record()
            torch.cuda.synchronize(device)
            latency_ms = start.elapsed_time(end) / repeats
        else:
            start_time = time.perf_counter()
            for _ in range(repeats):
                model(inputs)
            latency_ms = (time.perf_counter() - start_time) * 1000 / repeats
    fps = input_shape[0] * 1000 / latency_ms
    return latency_ms, fps


def get_validation_loader():
    data_path = ROOT / "IMX766" / "train_data"
    index_path = data_path / "train_data_idx.txt"
    if not data_path.exists() or not index_path.exists():
        return None, "validation data absent"
    try:
        from train_model.make_dataloader import Validset

        return torch.utils.data.DataLoader(Validset(), batch_size=1, shuffle=False), ""
    except (ImportError, OSError, AssertionError) as error:
        return None, f"validation loader unavailable: {error}"


def make_synthetic_raw(seed: int = 2023, train_count: int = 12, valid_count: int = 4):
    """Create deterministic smooth 4-channel targets with signal-dependent noise."""
    generator = torch.Generator().manual_seed(seed)
    count = train_count + valid_count
    low_resolution = torch.rand(count, 4, 8, 8, generator=generator)
    clean = F.interpolate(low_resolution, size=(64, 64), mode="bilinear", align_corners=False)
    gaussian = 0.03 * torch.randn(clean.shape, generator=generator)
    shot = 0.04 * torch.sqrt(clean.clamp_min(1e-4)) * torch.randn(clean.shape, generator=generator)
    noisy = (clean + gaussian + shot).clamp(0, 1)
    train = torch.utils.data.TensorDataset(noisy[:train_count], clean[:train_count])
    valid = torch.utils.data.TensorDataset(noisy[train_count:], clean[train_count:])
    return (
        torch.utils.data.DataLoader(train, batch_size=4, shuffle=False),
        torch.utils.data.DataLoader(valid, batch_size=1, shuffle=False),
    )


def fit_synthetic(model: torch.nn.Module, loader, device: torch.device, epochs: int) -> None:
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        for noisy, clean in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(noisy.to(device)), clean.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model = model.to(device).eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for noisy, clean in loader:
        prediction = model(noisy.to(device))
        target = clean.to(device)
        mse = torch.mean((prediction - target) ** 2)
        psnr_values.append(float(10 * torch.log10(1.0 / mse)))
        c1, c2 = 0.01**2, 0.03**2
        mean_x, mean_y = prediction.mean(), target.mean()
        std_x, std_y = prediction.std(), target.std()
        covariance = torch.mean((prediction - mean_x) * (target - mean_y))
        ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
            (mean_x**2 + mean_y**2 + c1) * (std_x**2 + std_y**2 + c2)
        )
        ssim_values.append(float(ssim))
    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)


def display_number(value: object, scale: float = 1.0, suffix: str = "", digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value) / scale:.{digits}f}{suffix}"


def write_readme_table(rows: list[dict[str, object]], markdown_path: Path, readme_path: Path) -> None:
    lines = [
        "| Model | Params | MACs | PSNR | SSIM | Latency | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {params} | {macs} | {psnr} | {ssim} | {latency} | {fps} |".format(
                model=row["model"],
                params=display_number(row["params"], 1e6, " M"),
                macs=display_number(row["macs"], 1e9, " G"),
                psnr=display_number(row["psnr"], suffix=" dB"),
                ssim=display_number(row["ssim"], digits=4),
                latency=display_number(row["latency_ms"], suffix=" ms"),
                fps=display_number(row["fps"], digits=1),
            )
        )
    table = "\n".join(lines)
    markdown_path.write_text(table + "\n", encoding="utf-8")

    start_marker = "<!-- LIGHTWEIGHT_RESULTS_START -->"
    end_marker = "<!-- LIGHTWEIGHT_RESULTS_END -->"
    readme = readme_path.read_text(encoding="utf-8")
    if start_marker in readme and end_marker in readme:
        before, rest = readme.split(start_marker, maxsplit=1)
        _, after = rest.split(end_marker, maxsplit=1)
        readme_path.write_text(
            f"{before}{start_marker}\n{table}\n{end_marker}{after}", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shape", nargs=4, type=int, default=(1, 4, 256, 256))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lightweight_ablation.csv"))
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--synthetic-eval",
        action="store_true",
        help="Fit every model on the same tiny deterministic synthetic RAW dataset",
    )
    parser.add_argument("--synthetic-epochs", type=int, default=2)
    args = parser.parse_args()
    input_shape = tuple(args.input_shape)
    if input_shape[1] != 4:
        parser.error("RAW input must have four channels")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    torch.manual_seed(2023)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2023)
    loader, loader_note = get_validation_loader()
    synthetic_train_loader = None
    if args.synthetic_eval and loader is None:
        synthetic_train_loader, loader = make_synthetic_raw()
        loader_note = "synthetic RAW smoke ablation; not an IMX766 camera benchmark"
    rows: list[dict[str, object]] = []
    for name in MODEL_REGISTRY:
        torch.manual_seed(2023)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2023)
        architectural = profile_model(name, input_shape)
        checkpoint = ROOT / CHECKPOINT_PATHS[name]
        model = create_model(name)
        quality_model = None
        status_parts: list[str] = []
        if checkpoint.exists():
            load_state_dict(model, checkpoint)
            quality_model = model
        elif synthetic_train_loader is not None:
            fit_synthetic(model, synthetic_train_loader, device, args.synthetic_epochs)
            quality_model = model
            status_parts.append(loader_note)
        else:
            status_parts.append("trained checkpoint absent")
        latency_ms, fps = measure_latency(model, input_shape, device, args.warmup, args.repeats)
        if loader is not None and quality_model is not None:
            psnr, ssim = evaluate(quality_model, loader, device)
        else:
            psnr, ssim = math.nan, math.nan
            if loader_note:
                status_parts.append(loader_note)
        rows.append(
            {
                "model": name,
                "params": architectural["params"],
                "macs": architectural["macs"],
                "psnr": psnr,
                "ssim": ssim,
                "latency_ms": latency_ms,
                "fps": fps,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "input_shape": "x".join(map(str, input_shape)),
                "checkpoint": str(CHECKPOINT_PATHS[name]) if checkpoint.exists() else "",
                "status": "; ".join(dict.fromkeys(status_parts)) or "complete",
            }
        )
        del model, quality_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_readme_table(rows, args.output.with_suffix(".md"), ROOT / "README.md")
    if not args.skip_plots:
        generate_plots(args.output, ROOT / "assets" / "lightweight")
    print(args.output)


if __name__ == "__main__":
    main()
