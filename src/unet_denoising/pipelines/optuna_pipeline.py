from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from unet_denoising.config import AppConfig
from unet_denoising.data.datasets import ExtendedPatchDataset, FixedValPatchDataset
from unet_denoising.data.io import DatasetBundle, build_dataset_bundle, list_tif_files
from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.models.unet import UNetDenoising
from unet_denoising.storage.google_drive import GoogleDriveStorage
from unet_denoising.training.engine import EpochMetrics, set_all_seeds, train
from unet_denoising.validation.runtime import ensure_dir_exists, ensure_writable_dir

logger = get_logger(__name__)



def _require_optuna_mlflow(enabled_mlflow: bool):
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("optuna is required for sweep-optuna. Install optuna.") from exc

    mlflow = None
    if enabled_mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
        except Exception as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("mlflow is enabled but not installed.") from exc

    return optuna, mlflow



def _select_score(metric_name: str, history: list[EpochMetrics]) -> float:
    if not history:
        return 0.0
    if metric_name == "val_ssim":
        return max(m.val_ssim for m in history)
    if metric_name == "val_psnr":
        return max(m.val_psnr for m in history)
    if metric_name == "val_loss":
        return min(m.val_loss for m in history)
    raise ValueError(f"Unsupported metric_name: {metric_name}")



def _best_epoch(metric_name: str, history: list[EpochMetrics]) -> int:
    if not history:
        return 0
    if metric_name == "val_loss":
        idx = int(np.argmin([m.val_loss for m in history]))
    elif metric_name == "val_psnr":
        idx = int(np.argmax([m.val_psnr for m in history]))
    elif metric_name == "val_ssim":
        idx = int(np.argmax([m.val_ssim for m in history]))
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")
    return idx + 1



def _prepare_bundle(cfg: AppConfig) -> tuple[str, str, DatasetBundle]:
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
    return train_noisy_dir, train_clean_dir, bundle



def run_optuna_sweep(cfg: AppConfig) -> None:
    if not cfg.optuna.enabled:
        raise ValueError("optuna.enabled is false. Set it true to run sweep-optuna.")

    ensure_writable_dir(cfg.storage.google_drive_root, "storage.root_dir")
    storage = GoogleDriveStorage(Path(cfg.storage.google_drive_root), cfg.storage.experiment_name)
    storage.ensure_dirs()
    run_dir = storage.create_run_dir("sweep_optuna")
    add_file_handler(logger, run_dir / "sweep_optuna.log")

    set_all_seeds(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, bundle = _prepare_bundle(cfg)

    optuna, mlflow = _require_optuna_mlflow(cfg.mlflow.enabled)
    if cfg.mlflow.enabled:
        if cfg.mlflow.tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    sampler = optuna.samplers.TPESampler(seed=cfg.optuna.sampler_seed)
    study = optuna.create_study(
        direction=cfg.optuna.direction,
        study_name=cfg.optuna.study_name,
        sampler=sampler,
    )

    def objective(trial) -> float:
        lr = trial.suggest_float("lr", cfg.optuna.lr_min, cfg.optuna.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", cfg.optuna.batch_size_choices)
        patch_size = trial.suggest_categorical("patch_size", cfg.optuna.patch_size_choices)
        border_size = trial.suggest_categorical("border_size", cfg.optuna.border_size_choices)
        crops_per_image = trial.suggest_categorical("crops_per_image", cfg.optuna.crops_per_image_choices)
        epochs = trial.suggest_categorical("epochs", cfg.optuna.epochs_choices)

        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_ckpt_dir = trial_dir / "checkpoints"
        trial_ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_ds = ExtendedPatchDataset(
            noisy_stack=bundle.noisy_norm,
            clean_stack=bundle.clean_norm,
            img_ids=bundle.train_ids,
            patch_size=patch_size,
            border_size=border_size,
            crops_per_image=crops_per_image,
        )
        val_ds = FixedValPatchDataset(
            noisy_stack=bundle.noisy_norm,
            clean_stack=bundle.clean_norm,
            img_ids=bundle.val_ids,
            patch_size=patch_size,
            border_size=border_size,
            stride=patch_size,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        model = UNetDenoising(in_channels=1).to(device)
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            noisy_norm=bundle.noisy_norm,
            clean_norm=bundle.clean_norm,
            val_img_ids=bundle.val_ids,
            epochs=epochs,
            lr=lr,
            patch_size=patch_size,
            border_size=border_size,
            device=device,
            ckpt_dir=trial_ckpt_dir,
            checkpoint_every=max(1, min(cfg.train.checkpoint_every, epochs)),
            logger=logger,
        )

        score = _select_score(cfg.optuna.metric_name, history)
        best_epoch = _best_epoch(cfg.optuna.metric_name, history)
        best_ckpt = trial_ckpt_dir / f"unet_halo_ep{best_epoch}.pth"

        trial_summary = {
            "trial": trial.number,
            "params": {
                "lr": lr,
                "batch_size": batch_size,
                "patch_size": patch_size,
                "border_size": border_size,
                "crops_per_image": crops_per_image,
                "epochs": epochs,
            },
            "metric_name": cfg.optuna.metric_name,
            "score": score,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_ckpt),
            "history": [asdict(h) for h in history],
        }
        (trial_dir / "summary.json").write_text(json.dumps(trial_summary, indent=2))

        trial.set_user_attr("best_checkpoint", str(best_ckpt))
        trial.set_user_attr("summary_path", str(trial_dir / "summary.json"))

        if cfg.mlflow.enabled:
            with mlflow.start_run(run_name=f"{cfg.mlflow.run_name_prefix}_trial_{trial.number}", nested=True):
                mlflow.log_params(trial_summary["params"])
                mlflow.log_param("metric_name", cfg.optuna.metric_name)
                for idx, h in enumerate(history, start=1):
                    mlflow.log_metric("train_loss", h.train_loss, step=idx)
                    mlflow.log_metric("val_loss", h.val_loss, step=idx)
                    mlflow.log_metric("val_psnr", h.val_psnr, step=idx)
                    mlflow.log_metric("val_ssim", h.val_ssim, step=idx)
                mlflow.log_metric(f"best_{cfg.optuna.metric_name}", float(score))
                if best_ckpt.exists():
                    mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")
                mlflow.log_artifact(str(trial_dir / "summary.json"), artifact_path="metadata")

        return float(score)

    study.optimize(objective, n_trials=cfg.optuna.n_trials, timeout=cfg.optuna.timeout_sec)

    valid_trials = [t for t in study.trials if t.value is not None]
    reverse = cfg.optuna.direction == "maximize"
    top_trials = sorted(valid_trials, key=lambda t: float(t.value), reverse=reverse)[: cfg.optuna.top_k]

    topk_dir = run_dir / "top_k"
    topk_dir.mkdir(parents=True, exist_ok=True)

    topk_summary: list[dict[str, Any]] = []
    for rank, t in enumerate(top_trials, start=1):
        rank_summary = {
            "rank": rank,
            "trial": t.number,
            "score": float(t.value),
            "params": t.params,
            "best_checkpoint": t.user_attrs.get("best_checkpoint"),
            "summary_path": t.user_attrs.get("summary_path"),
        }
        topk_summary.append(rank_summary)

        if cfg.mlflow.enabled:
            with mlflow.start_run(run_name=f"{cfg.mlflow.run_name_prefix}_topk_rank_{rank}"):
                mlflow.log_params({f"ranked_{k}": v for k, v in t.params.items()})
                mlflow.log_metric(f"topk_{cfg.optuna.metric_name}", float(t.value))
                mlflow.set_tag("rank", rank)
                ckpt = t.user_attrs.get("best_checkpoint")
                if ckpt and Path(ckpt).exists():
                    mlflow.log_artifact(ckpt, artifact_path="topk_checkpoints")
                summary_path = t.user_attrs.get("summary_path")
                if summary_path and Path(summary_path).exists():
                    mlflow.log_artifact(summary_path, artifact_path="topk_metadata")

    (topk_dir / "summary.json").write_text(json.dumps(topk_summary, indent=2))
    logger.info("Optuna sweep complete. top_k=%d metric=%s run_dir=%s", len(top_trials), cfg.optuna.metric_name, run_dir)
