from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from unet_denoising.config import AppConfig
from unet_denoising.data.datasets import ExtendedPatchDataset, FixedValPatchDataset
from unet_denoising.data.io import build_dataset_bundle, list_tif_files
from unet_denoising.exceptions import QualityGateError
from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.models.unet import UNetDenoising
from unet_denoising.storage.google_drive import GoogleDriveStorage
from unet_denoising.training.engine import set_all_seeds, train
from unet_denoising.validation.runtime import ensure_dir_exists, ensure_writable_dir


logger = get_logger(__name__)


def _check_quality_gate(cfg: AppConfig, history: list) -> None:
    if not cfg.quality_gate.enabled:
        return
    if not history:
        raise QualityGateError("Quality gate enabled but training history is empty.")

    last = history[-1]
    failures: list[str] = []
    if cfg.quality_gate.min_val_psnr is not None and last.val_psnr < cfg.quality_gate.min_val_psnr:
        failures.append(
            f"val_psnr {last.val_psnr:.4f} < min_val_psnr {cfg.quality_gate.min_val_psnr:.4f}"
        )
    if cfg.quality_gate.min_val_ssim is not None and last.val_ssim < cfg.quality_gate.min_val_ssim:
        failures.append(
            f"val_ssim {last.val_ssim:.4f} < min_val_ssim {cfg.quality_gate.min_val_ssim:.4f}"
        )
    if cfg.quality_gate.max_val_loss is not None and last.val_loss > cfg.quality_gate.max_val_loss:
        failures.append(
            f"val_loss {last.val_loss:.6f} > max_val_loss {cfg.quality_gate.max_val_loss:.6f}"
        )
    if failures:
        raise QualityGateError("Quality gate failed: " + "; ".join(failures))


def run_train(cfg: AppConfig) -> None:
    if cfg.s3 is not None and cfg.s3.auto_pull:
        logger.info("S3 auto_pull enabled. Pulling dataset/artifacts before training.")
        from unet_denoising.pipelines.s3_pipeline import run_s3_pull

        run_s3_pull(cfg)

    ensure_writable_dir(cfg.storage.google_drive_root, "storage.google_drive_root")
    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("train")
    add_file_handler(logger, run_dir / "train.log")
    logger.info("Run directory: %s", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(cfg.train.seed)

    train_noisy_dir = cfg.paths.noisy_train_dir
    train_clean_dir = cfg.paths.gt_train_dir
    ensure_dir_exists(train_clean_dir, "paths.gt_train_dir")
    if not list_tif_files(train_noisy_dir):
        logger.warning(
            "No training TIFF files in %s. Falling back to %s",
            train_noisy_dir,
            cfg.paths.noisy_val_dir,
        )
        train_noisy_dir = cfg.paths.noisy_val_dir
        train_clean_dir = cfg.paths.gt_val_dir
    ensure_dir_exists(train_noisy_dir, "effective train noisy dir")
    ensure_dir_exists(train_clean_dir, "effective train clean dir")

    bundle = build_dataset_bundle(
        noisy_dir=train_noisy_dir,
        clean_dir=train_clean_dir,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.train.seed,
    )

    train_ds = ExtendedPatchDataset(
        noisy_stack=bundle.noisy_norm,
        clean_stack=bundle.clean_norm,
        img_ids=bundle.train_ids,
        patch_size=cfg.data.patch_size,
        border_size=cfg.data.border_size,
        crops_per_image=cfg.data.crops_per_image,
    )
    val_ds = FixedValPatchDataset(
        noisy_stack=bundle.noisy_norm,
        clean_stack=bundle.clean_norm,
        img_ids=bundle.val_ids,
        patch_size=cfg.data.patch_size,
        border_size=cfg.data.border_size,
        stride=cfg.data.patch_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = UNetDenoising(in_channels=1).to(device)

    logger.info("Starting training with %d train images and %d val images", len(bundle.train_ids), len(bundle.val_ids))
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        noisy_norm=bundle.noisy_norm,
        clean_norm=bundle.clean_norm,
        val_img_ids=bundle.val_ids,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        patch_size=cfg.data.patch_size,
        border_size=cfg.data.border_size,
        device=device,
        ckpt_dir=storage.checkpoints_dir,
        checkpoint_every=cfg.train.checkpoint_every,
        logger=logger,
    )

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps([m.__dict__ for m in history], indent=2))

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot([m.train_loss for m in history])
    plt.title("Train Loss")
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot([m.val_loss for m in history])
    plt.title("Val Loss")
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot([m.val_psnr for m in history])
    plt.title("Val PSNR")
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot([m.val_ssim for m in history])
    plt.title("Val SSIM")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=300)

    stats_path = storage.experiment_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(bundle.stats.__dict__, indent=2))
    run_summary = {
        "train_noisy_dir": train_noisy_dir,
        "train_clean_dir": train_clean_dir,
        "train_images": len(bundle.train_ids),
        "val_images": len(bundle.val_ids),
        "history_path": str(history_path),
        "training_curves": str(run_dir / "training_curves.png"),
        "checkpoints_dir": str(storage.checkpoints_dir),
        "normalization_stats": str(stats_path),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    _check_quality_gate(cfg, history)
    logger.info("Training completed. Artifacts: %s", run_dir)
