from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_ssim(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    dr = float(gt.max() - gt.min()) or 1.0
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=dr))
    ssim = float(structural_similarity(gt, pred, data_range=dr))
    return psnr, ssim
