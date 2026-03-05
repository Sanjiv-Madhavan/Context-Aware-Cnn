from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

from unet_denoising.config import (
    AppConfig,
    DataConfig,
    InferenceConfig,
    PathsConfig,
    S3Config,
    StorageConfig,
    TrainConfig,
)

# Headless-safe plotting backend for pipeline tests that import matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


def make_config(root: Path, with_s3: bool = True, auto_pull: bool = False) -> AppConfig:
    data_root = root / "data"
    paths = PathsConfig(
        noisy_train_dir=str(data_root / "small_images/noisy"),
        gt_train_dir=str(data_root / "small_images/gt"),
        noisy_val_dir=str(data_root / "large_images/noisy"),
        gt_val_dir=str(data_root / "large_images/gt"),
        noisy_infer_dir=str(data_root / "test_images/noisy"),
        gt_infer_dir=str(data_root / "test_images/gt"),
    )
    storage = StorageConfig(root_dir=str(root / "artifacts"), experiment_name="exp")
    data = DataConfig(patch_size=4, border_size=1, crops_per_image=2, val_ratio=0.5, num_workers=0)
    train = TrainConfig(seed=1, epochs=1, batch_size=1, lr=1e-3, checkpoint_every=1)
    inference = InferenceConfig(checkpoint_name="unet_halo_ep1.pth")

    s3 = None
    if with_s3:
        s3 = S3Config(
            endpoint_url="http://localhost:4566",
            dataset_bucket="dataset-bucket",
            artifacts_bucket="artifacts-bucket",
            dataset_prefix="",
            artifacts_prefix="artifacts",
            verify_ssl=False,
            auto_pull=auto_pull,
        )

    return AppConfig(paths=paths, storage=storage, data=data, train=train, inference=inference, s3=s3)


def seed_tif_tree(cfg: AppConfig, value: float = 1.0) -> None:
    dirs = [
        cfg.paths.noisy_train_dir,
        cfg.paths.gt_train_dir,
        cfg.paths.noisy_val_dir,
        cfg.paths.gt_val_dir,
        cfg.paths.noisy_infer_dir,
        cfg.paths.gt_infer_dir,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    arr = np.full((8, 8), value, dtype=np.float32)
    tifffile.imwrite(Path(cfg.paths.noisy_train_dir) / "a.tif", arr)
    tifffile.imwrite(Path(cfg.paths.gt_train_dir) / "a.tif", arr)
    tifffile.imwrite(Path(cfg.paths.noisy_val_dir) / "a.tif", arr)
    tifffile.imwrite(Path(cfg.paths.gt_val_dir) / "a.tif", arr)
    tifffile.imwrite(Path(cfg.paths.noisy_infer_dir) / "a.tif", arr)
    tifffile.imwrite(Path(cfg.paths.gt_infer_dir) / "a.tif", arr)
