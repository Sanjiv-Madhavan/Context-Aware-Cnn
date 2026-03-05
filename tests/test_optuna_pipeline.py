from __future__ import annotations

import pytest

from unet_denoising.training.engine import EpochMetrics
from unet_denoising.pipelines.optuna_pipeline import _best_epoch, _select_score


def test_select_score_variants():
    history = [
        EpochMetrics(train_loss=1.0, val_loss=0.7, val_psnr=20.0, val_ssim=0.70),
        EpochMetrics(train_loss=0.8, val_loss=0.6, val_psnr=21.0, val_ssim=0.75),
    ]
    assert _select_score("val_ssim", history) == pytest.approx(0.75)
    assert _select_score("val_psnr", history) == pytest.approx(21.0)
    assert _select_score("val_loss", history) == pytest.approx(0.6)


def test_best_epoch_variants():
    history = [
        EpochMetrics(train_loss=1.0, val_loss=0.7, val_psnr=20.0, val_ssim=0.70),
        EpochMetrics(train_loss=0.8, val_loss=0.6, val_psnr=21.0, val_ssim=0.75),
        EpochMetrics(train_loss=0.7, val_loss=0.65, val_psnr=19.0, val_ssim=0.73),
    ]
    assert _best_epoch("val_ssim", history) == 2
    assert _best_epoch("val_psnr", history) == 2
    assert _best_epoch("val_loss", history) == 2


def test_select_score_bad_metric():
    history = [EpochMetrics(train_loss=1.0, val_loss=0.7, val_psnr=20.0, val_ssim=0.70)]
    with pytest.raises(ValueError):
        _select_score("x", history)
