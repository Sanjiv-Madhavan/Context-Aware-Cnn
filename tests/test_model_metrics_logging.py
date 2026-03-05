from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from unet_denoising.logging_utils import add_file_handler, get_logger
from unet_denoising.metrics import psnr_ssim
from unet_denoising.models.unet import UNetDenoising


def test_model_forward_shape():
    model = UNetDenoising(1)
    x = torch.randn(1, 1, 16, 16)
    y = model(x)
    assert y.shape == x.shape


def test_metrics_identical_arrays():
    arr = np.ones((8, 8), dtype=np.float32)
    p, s = psnr_ssim(arr, arr)
    assert p > 50
    assert s > 0.99


def test_file_logger_handler_dedup(tmp_path):
    logger = get_logger("test.logger")
    log_file = Path(tmp_path) / "x.log"
    add_file_handler(logger, log_file)
    add_file_handler(logger, log_file)
    logger.info("hello")
    txt = log_file.read_text()
    assert "hello" in txt
