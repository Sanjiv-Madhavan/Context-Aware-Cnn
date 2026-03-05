from __future__ import annotations

import numpy as np
import torch
from torch import nn

from unet_denoising.data.datasets import reflect_pad_crop


@torch.no_grad()
def raster_infer_halo(
    noisy_img: np.ndarray,
    model: nn.Module,
    patch_size: int,
    border_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    h, w = noisy_img.shape
    outer_size = patch_size + 2 * border_size

    out = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, patch_size):
        core_h = min(patch_size, h - y)
        for x in range(0, w, patch_size):
            core_w = min(patch_size, w - x)

            oy, ox = y - border_size, x - border_size
            inp = reflect_pad_crop(noisy_img, oy, ox, outer_size, outer_size).copy()

            if y > 0:
                top_y0 = max(0, y - border_size)
                top_y1 = y
                top_x0 = max(0, x - border_size)
                top_x1 = min(w, x + core_w + border_size)
                pred_top = out[top_y0:top_y1, top_x0:top_x1]

                dst_y0, dst_y1 = 0, top_y1 - (y - border_size)
                dst_x0 = top_x0 - ox
                dst_x1 = dst_x0 + pred_top.shape[1]

                dst_x0 = max(0, dst_x0)
                dst_x1 = min(outer_size, dst_x1)
                pred_top = pred_top[:, : max(0, dst_x1 - dst_x0)]
                if pred_top.size > 0:
                    inp[dst_y0:dst_y1, dst_x0:dst_x1] = pred_top

            if x > 0:
                left_x0 = max(0, x - border_size)
                left_x1 = x
                left_y0 = max(0, y - border_size)
                left_y1 = min(h, y + core_h + border_size)
                pred_left = out[left_y0:left_y1, left_x0:left_x1]

                dst_x0, dst_x1 = 0, left_x1 - (x - border_size)
                dst_y0 = left_y0 - oy
                dst_y1 = dst_y0 + pred_left.shape[0]

                dst_y0 = max(0, dst_y0)
                dst_y1 = min(outer_size, dst_y1)
                pred_left = pred_left[: max(0, dst_y1 - dst_y0), :]
                if pred_left.size > 0:
                    inp[dst_y0:dst_y1, dst_x0:dst_x1] = pred_left

            tin = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
            pred_outer = model(tin).squeeze(0)
            pred_core = pred_outer[:, border_size : border_size + core_h, border_size : border_size + core_w]
            out[y : y + core_h, x : x + core_w] = pred_core.squeeze(0).cpu().numpy()

    return out
