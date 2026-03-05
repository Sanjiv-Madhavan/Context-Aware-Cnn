from __future__ import annotations

import random
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_denoising.inference.halo import raster_infer_halo
from unet_denoising.metrics import psnr_ssim


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_psnr: float
    val_ssim: float


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def crop_core(pred_outer: torch.Tensor, patch_size: int, border_size: int) -> torch.Tensor:
    return pred_outer[
        :,
        :,
        border_size : border_size + patch_size,
        border_size : border_size + patch_size,
    ]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    noisy_norm: np.ndarray,
    clean_norm: np.ndarray,
    val_img_ids: list[int],
    epochs: int,
    lr: float,
    patch_size: int,
    border_size: int,
    device: torch.device,
    ckpt_dir: Path,
    checkpoint_every: int,
    logger: logging.Logger | None = None,
) -> list[EpochMetrics]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x_outer, y_core in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x_outer = x_outer.to(device)
            y_core = y_core.to(device)

            pred_outer = model(x_outer)
            pred_core = crop_core(pred_outer, patch_size=patch_size, border_size=border_size)
            loss = criterion(pred_core, y_core)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for x_outer, y_core in val_loader:
                x_outer = x_outer.to(device)
                y_core = y_core.to(device)
                pred_outer = model(x_outer)
                pred_core = crop_core(pred_outer, patch_size=patch_size, border_size=border_size)
                val_running += float(criterion(pred_core, y_core).item())

            avg_val_loss = val_running / max(1, len(val_loader))

            psnr_vals: list[float] = []
            ssim_vals: list[float] = []
            for vid in val_img_ids:
                pred = raster_infer_halo(noisy_norm[vid], model, patch_size, border_size, device)
                p, s = psnr_ssim(clean_norm[vid], pred)
                psnr_vals.append(p)
                ssim_vals.append(s)

            mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
            mean_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0

        history.append(EpochMetrics(avg_train_loss, avg_val_loss, mean_psnr, mean_ssim))
        if logger is not None:
            logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f val_psnr=%.4f val_ssim=%.4f",
                epoch,
                avg_train_loss,
                avg_val_loss,
                mean_psnr,
                mean_ssim,
            )

        if epoch % checkpoint_every == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"unet_halo_ep{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            if logger is not None:
                logger.info("Saved checkpoint: %s", ckpt_path)

    return history
