from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile

from conftest import make_config, seed_tif_tree
from unet_denoising.data.io import DatasetBundle, NormalizationStats
from unet_denoising.exceptions import QualityGateError
from unet_denoising.pipelines import download_pipeline, infer_pipeline, train_pipeline
from unet_denoising.training.engine import EpochMetrics


def test_download_pipeline_happy_path(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, with_s3=False)
    cfg.dataset = type("D", (), {
        "download_url": "http://x",
        "zip_path": str(tmp_path / "raw/a.zip"),
        "extract_dir": str(tmp_path / "raw/extract"),
        "overwrite": False,
        "verify_ssl": False,
        "ca_cert_path": None,
    })()

    monkeypatch.setattr(download_pipeline, "download_file", lambda *a, **k: Path(cfg.dataset.zip_path).parent.mkdir(parents=True, exist_ok=True) or Path(cfg.dataset.zip_path).write_text("z"))
    monkeypatch.setattr(download_pipeline, "extract_zip", lambda *a, **k: Path(cfg.dataset.extract_dir).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(download_pipeline, "organize_dataset", lambda *a, **k: {"train_noisy": 1})

    download_pipeline.run_download(cfg)
    runs = list((Path(cfg.storage.root_dir) / "experiments/exp/runs").glob("download_*"))
    assert runs


def test_train_pipeline_happy_path(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, with_s3=False)
    seed_tif_tree(cfg)

    bundle = DatasetBundle(
        noisy_norm=np.zeros((1, 8, 8), dtype=np.float32),
        clean_norm=np.zeros((1, 8, 8), dtype=np.float32),
        train_ids=[0],
        val_ids=[0],
        stats=NormalizationStats(0.0, 1.0, 0.0, 1.0),
        file_names=["a.tif"],
    )

    monkeypatch.setattr(train_pipeline, "build_dataset_bundle", lambda **kwargs: bundle)
    monkeypatch.setattr(train_pipeline, "train", lambda **kwargs: [EpochMetrics(1.0, 1.0, 1.0, 1.0)])

    train_pipeline.run_train(cfg)
    exp_dir = Path(cfg.storage.root_dir) / "experiments/exp"
    assert (exp_dir / "normalization_stats.json").exists()


def test_infer_pipeline_fallback_stats(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, with_s3=False)
    seed_tif_tree(cfg, value=2.0)

    # checkpoint exists
    ckpt_dir = Path(cfg.storage.root_dir) / "experiments/exp/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from unet_denoising.models.unet import UNetDenoising

    model = UNetDenoising(in_channels=1)
    torch.save(model.state_dict(), ckpt_dir / cfg.inference.checkpoint_name)

    monkeypatch.setattr(infer_pipeline, "raster_infer_halo", lambda noisy_img, **kwargs: noisy_img)

    infer_pipeline.run_infer(cfg)
    run_dirs = list((Path(cfg.storage.root_dir) / "experiments/exp/runs").glob("infer_*"))
    assert run_dirs
    outputs = list((run_dirs[-1] / "outputs").glob("denoised_*.tif"))
    assert outputs

    # output is written as uint16
    out_img = tifffile.imread(outputs[0])
    assert out_img.dtype == np.uint16


def test_train_quality_gate_failure(tmp_path):
    cfg = make_config(tmp_path, with_s3=False)
    cfg.quality_gate.enabled = True
    cfg.quality_gate.min_val_psnr = 10.0
    history = [EpochMetrics(train_loss=1.0, val_loss=1.0, val_psnr=1.0, val_ssim=0.2)]
    with pytest.raises(QualityGateError):
        train_pipeline._check_quality_gate(cfg, history)


def test_train_quality_gate_pass(tmp_path):
    cfg = make_config(tmp_path, with_s3=False)
    cfg.quality_gate.enabled = True
    cfg.quality_gate.min_val_psnr = 1.0
    cfg.quality_gate.min_val_ssim = 0.1
    cfg.quality_gate.max_val_loss = 2.0
    history = [EpochMetrics(train_loss=1.0, val_loss=1.0, val_psnr=2.0, val_ssim=0.5)]
    train_pipeline._check_quality_gate(cfg, history)
