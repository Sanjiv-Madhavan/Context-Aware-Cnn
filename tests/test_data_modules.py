from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from unet_denoising.data.datasets import (
    ExtendedPatchDataset,
    FixedValPatchDataset,
    build_training_outer,
    reflect_pad_crop,
)
from unet_denoising.data.io import build_dataset_bundle, load_image_pairs, split_ids


def test_io_load_and_bundle(tmp_path):
    noisy = tmp_path / "noisy"
    clean = tmp_path / "clean"
    noisy.mkdir(); clean.mkdir()

    n = np.ones((4, 4), dtype=np.float32)
    c = np.ones((4, 4), dtype=np.float32) * 2
    tifffile.imwrite(noisy / "a.tif", n)
    tifffile.imwrite(clean / "a.tif", c)

    noisy_stack, clean_stack, names = load_image_pairs(str(noisy), str(clean))
    assert noisy_stack.shape == (1, 4, 4)
    assert clean_stack.shape == (1, 4, 4)
    assert names == ["a.tif"]

    bundle = build_dataset_bundle(str(noisy), str(clean), val_ratio=0.5, seed=1)
    assert bundle.noisy_norm.shape == (1, 4, 4)
    assert bundle.stats.noisy_std >= 0.0


def test_io_mismatch_raises(tmp_path):
    noisy = tmp_path / "n"; clean = tmp_path / "c"
    noisy.mkdir(); clean.mkdir()
    tifffile.imwrite(noisy / "a.tif", np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        load_image_pairs(str(noisy), str(clean))


def test_split_ids():
    train_ids, val_ids = split_ids(10, 0.2, seed=42)
    assert len(val_ids) >= 1
    assert set(train_ids).isdisjoint(val_ids)


def test_datasets_and_padding():
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    crop = reflect_pad_crop(img, -1, -1, 4, 4)
    assert crop.shape == (4, 4)

    inp, gt = build_training_outer(img, img, core_y=0, core_x=0, patch_size=2, border_size=1)
    assert inp.shape == (4, 4)
    assert gt.shape == (2, 2)

    noisy = np.stack([img, img], axis=0)
    clean = np.stack([img, img], axis=0)
    ds = ExtendedPatchDataset(noisy, clean, [0, 1], patch_size=2, border_size=1, crops_per_image=2)
    x, y = ds[0]
    assert x.shape == (1, 4, 4)
    assert y.shape == (1, 2, 2)

    vds = FixedValPatchDataset(noisy, clean, [0], patch_size=2, border_size=1)
    vx, vy = vds[0]
    assert vx.shape[0] == 1
    assert vy.shape[0] == 1
