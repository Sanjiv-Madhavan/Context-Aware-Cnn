from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset


def reflect_pad_crop(img: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    height, width = img.shape
    y1, x1 = y0 + h, x0 + w
    pt, pl = max(0, -y0), max(0, -x0)
    pb, pr = max(0, y1 - height), max(0, x1 - width)

    if pt or pb or pl or pr:
        padded = np.pad(img, ((pt, pb), (pl, pr)), mode="reflect")
        return padded[y0 + pt : y0 + pt + h, x0 + pl : x0 + pl + w]
    return img[y0:y1, x0:x1]


def build_training_outer(
    noisy_img: np.ndarray,
    clean_img: np.ndarray,
    core_y: int,
    core_x: int,
    patch_size: int,
    border_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer_size = patch_size + 2 * border_size
    oy, ox = core_y - border_size, core_x - border_size

    noisy_outer = reflect_pad_crop(noisy_img, oy, ox, outer_size, outer_size)
    gt_outer = reflect_pad_crop(clean_img, oy, ox, outer_size, outer_size)

    inp = noisy_outer.copy()
    inp[:border_size, :] = gt_outer[:border_size, :]
    inp[:, :border_size] = gt_outer[:, :border_size]

    gt_core = clean_img[core_y : core_y + patch_size, core_x : core_x + patch_size]
    return inp.astype(np.float32), gt_core.astype(np.float32)


class ExtendedPatchDataset(Dataset):
    def __init__(
        self,
        noisy_stack: np.ndarray,
        clean_stack: np.ndarray,
        img_ids: list[int],
        patch_size: int,
        border_size: int,
        crops_per_image: int,
    ) -> None:
        self.noisy = noisy_stack
        self.clean = clean_stack
        self.img_ids = list(img_ids)
        self.patch_size = patch_size
        self.border_size = border_size
        self.height = noisy_stack.shape[1]
        self.width = noisy_stack.shape[2]
        self.total = len(self.img_ids) * crops_per_image

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        del idx
        img_id = random.choice(self.img_ids)
        max_y = self.height - self.patch_size
        max_x = self.width - self.patch_size

        y = random.randint(0, max_y)
        x = random.randint(0, max_x)

        inp, gt = build_training_outer(
            noisy_img=self.noisy[img_id],
            clean_img=self.clean[img_id],
            core_y=y,
            core_x=x,
            patch_size=self.patch_size,
            border_size=self.border_size,
        )

        x_tensor = torch.from_numpy(inp).unsqueeze(0)
        y_tensor = torch.from_numpy(gt).unsqueeze(0)

        if random.random() > 0.5:
            x_tensor = torch.flip(x_tensor, dims=[2])
            y_tensor = torch.flip(y_tensor, dims=[2])
        if random.random() > 0.5:
            x_tensor = torch.flip(x_tensor, dims=[1])
            y_tensor = torch.flip(y_tensor, dims=[1])
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            x_tensor = torch.rot90(x_tensor, k=k, dims=[1, 2])
            y_tensor = torch.rot90(y_tensor, k=k, dims=[1, 2])

        return x_tensor, y_tensor


class FixedValPatchDataset(Dataset):
    def __init__(
        self,
        noisy_stack: np.ndarray,
        clean_stack: np.ndarray,
        img_ids: list[int],
        patch_size: int,
        border_size: int,
        stride: int | None = None,
    ) -> None:
        self.noisy = noisy_stack
        self.clean = clean_stack
        self.patch_size = patch_size
        self.border_size = border_size
        self.height = noisy_stack.shape[1]
        self.width = noisy_stack.shape[2]
        self.stride = patch_size if stride is None else stride

        self.coords: list[tuple[int, int, int]] = []
        ys = list(range(0, max(1, self.height - self.patch_size + 1), self.stride))
        xs = list(range(0, max(1, self.width - self.patch_size + 1), self.stride))

        for img_id in img_ids:
            for y in ys:
                y0 = min(y, self.height - self.patch_size)
                for x in xs:
                    x0 = min(x, self.width - self.patch_size)
                    self.coords.append((img_id, y0, x0))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id, y, x = self.coords[idx]
        inp, gt = build_training_outer(
            noisy_img=self.noisy[img_id],
            clean_img=self.clean[img_id],
            core_y=y,
            core_x=x,
            patch_size=self.patch_size,
            border_size=self.border_size,
        )
        return torch.from_numpy(inp).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)
