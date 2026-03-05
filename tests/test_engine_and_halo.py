from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from unet_denoising.inference.halo import raster_infer_halo
from unet_denoising.training import engine


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x


def test_raster_infer_halo_identity():
    img = np.ones((8, 8), dtype=np.float32)
    out = raster_infer_halo(img, IdentityModel(), patch_size=4, border_size=1, device=torch.device("cpu"))
    assert out.shape == img.shape


def test_train_engine_saves_checkpoint(tmp_path, monkeypatch):
    x = torch.randn(2, 1, 4, 4)
    y = torch.randn(2, 1, 2, 2)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=1)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=1)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(1, 1, 1)

        def forward(self, xx):
            return self.c(xx)

    monkeypatch.setattr(engine, "raster_infer_halo", lambda *a, **k: np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(engine, "psnr_ssim", lambda *a, **k: (1.0, 0.5))

    history = engine.train(
        model=Tiny(),
        train_loader=train_loader,
        val_loader=val_loader,
        noisy_norm=np.zeros((1, 4, 4), dtype=np.float32),
        clean_norm=np.zeros((1, 4, 4), dtype=np.float32),
        val_img_ids=[0],
        epochs=1,
        lr=1e-3,
        patch_size=2,
        border_size=1,
        device=torch.device("cpu"),
        ckpt_dir=Path(tmp_path),
        checkpoint_every=1,
    )
    assert len(history) == 1
    assert (Path(tmp_path) / "unet_halo_ep1.pth").exists()
