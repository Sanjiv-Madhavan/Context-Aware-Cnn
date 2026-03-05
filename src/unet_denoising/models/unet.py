from __future__ import annotations

import torch
from torch import nn


class UNetDenoising(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()

        def conv_block(in_c: int, out_c: int, dropout: float = 0.0) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.c1 = conv_block(in_channels, 16, 0.1)
        self.c2 = conv_block(16, 32, 0.1)
        self.c3 = conv_block(32, 64, 0.2)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c4 = conv_block(64, 32, 0.1)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c5 = conv_block(32, 16, 0.1)

        self.out = nn.Conv2d(16, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.c1(x)
        p1 = self.pool(c1)

        c2 = self.c2(p1)
        p2 = self.pool(c2)

        c3 = self.c3(p2)

        u4 = self.up4(c3)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.c4(u4)

        u5 = self.up5(c4)
        u5 = torch.cat([u5, c1], dim=1)
        c5 = self.c5(u5)

        return self.out(c5)
