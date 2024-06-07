import torch
from torch import nn
from torch.nn import functional as F

from segmentation.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm: bool = True,
            attention_type=None
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        conv2 = md.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks: int = 5,
            use_batchnorm: bool = True,
            attention_type=None,
            center: bool = False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(encoder_channels)} blocks"
            )

        # Remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]

        # Reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # Combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
