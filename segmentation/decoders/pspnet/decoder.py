import torch
from torch import nn
from torch.nn import functional as F

from segmentation.base import modules


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_batchnorm=True):
        super().__init__()
        if pool_size == 1:
            use_batchnorm = False  # 1x1 shape not supported
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_batchnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_batchnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(
                in_channels,
                in_channels // len(sizes),
                size,
                use_batchnorm=use_batchnorm
            )
            for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            use_batchnorm=True,
            out_channels=512,
            dropout=0.2
    ):
        super().__init__()
        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_batchnorm=use_batchnorm
        )

        self.conv = modules.Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x

