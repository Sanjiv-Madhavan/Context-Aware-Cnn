from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from unet_denoising.config import AppConfig
from unet_denoising.data.io import list_tif_files
from unet_denoising.inference.halo import raster_infer_halo
from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.models.unet import UNetDenoising
from unet_denoising.storage.google_drive import GoogleDriveStorage
from unet_denoising.validation.runtime import ensure_dir_exists, ensure_file_exists, ensure_writable_dir


logger = get_logger(__name__)


def _compute_cuboid_stats(noisy_files: list[str]) -> tuple[float, float]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for noisy_file in noisy_files:
        noisy = np.squeeze(tifffile.imread(noisy_file).astype(np.float32))
        total_sum += float(noisy.sum())
        total_sumsq += float(np.square(noisy).sum())
        total_count += int(noisy.size)

    if total_count == 0:
        return 0.0, 1.0

    mean = total_sum / total_count
    var = max(0.0, (total_sumsq / total_count) - (mean * mean))
    std = float(np.sqrt(var))
    return float(mean), float(std)


def run_infer(cfg: AppConfig) -> None:
    if cfg.s3 is not None and cfg.s3.auto_pull:
        logger.info("S3 auto_pull enabled. Pulling dataset/artifacts before inference.")
        from unet_denoising.pipelines.s3_pipeline import run_s3_pull

        run_s3_pull(cfg)

    ensure_writable_dir(cfg.storage.google_drive_root, "storage.google_drive_root")
    ensure_dir_exists(cfg.paths.noisy_infer_dir, "paths.noisy_infer_dir")
    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("infer")
    run_outputs_dir = run_dir / "outputs"
    run_outputs_dir.mkdir(parents=True, exist_ok=True)
    add_file_handler(logger, run_dir / "infer.log")
    logger.info("Run directory: %s", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoising(in_channels=1).to(device)

    ckpt_path = storage.checkpoints_dir / cfg.inference.checkpoint_name
    ensure_file_exists(ckpt_path, "inference checkpoint")
    logger.info("Loading checkpoint: %s", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    noisy_files = list_tif_files(cfg.paths.noisy_infer_dir)
    if not noisy_files:
        raise ValueError(f"No .tif files found in {cfg.paths.noisy_infer_dir}")

    stats_path = storage.experiment_dir / "normalization_stats.json"
    if stats_path.exists():
        stats = json.loads(ensure_file_exists(stats_path, "normalization stats").read_text())
        noisy_mean = float(stats["noisy_mean"])
        noisy_std = float(stats["noisy_std"])
        logger.info("Using normalization stats from %s", stats_path)
    else:
        noisy_mean, noisy_std = _compute_cuboid_stats(noisy_files)
        logger.warning(
            "normalization_stats.json not found. Falling back to inference-cuboid stats "
            "(mean=%.6f, std=%.6f).",
            noisy_mean,
            noisy_std,
        )

    for noisy_file in noisy_files:
        name = Path(noisy_file).name
        noisy = np.squeeze(tifffile.imread(noisy_file).astype(np.float32))
        noisy_norm = (noisy - noisy_mean) / (noisy_std + 1e-8)

        den_norm = raster_infer_halo(
            noisy_img=noisy_norm,
            model=model,
            patch_size=cfg.data.patch_size,
            border_size=cfg.data.border_size,
            device=device,
        )
        den = den_norm * (noisy_std + 1e-8) + noisy_mean

        out_path = run_outputs_dir / f"denoised_{name}"
        tifffile.imwrite(
            out_path,
            den.astype(np.uint16),
            photometric="minisblack",
            metadata={"axes": "YX"},
        )
        logger.info("Wrote output: %s", out_path)

    run_summary = {
        "checkpoint": str(ckpt_path),
        "inputs_dir": cfg.paths.noisy_infer_dir,
        "outputs_dir": str(run_outputs_dir),
        "num_images": len(noisy_files),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    logger.info("Inference completed. Outputs in %s", run_outputs_dir)
