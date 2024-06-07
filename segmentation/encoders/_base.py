import torch
from torch import nn

from . import _utils as utils


class EncoderMixin:
    """
    Mixin for encoders
    """

    _output_stride = 32

    def __init__(self):
        self._out_channels = None
        self._in_channels = None

    @property
    def out_channels(self):
        return self._out_channels[: self._depth+1]

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels, pretrained=True):
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])
        utils.patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)

    def get_stages(self):
        raise NotImplementedError

    def make_dilated(self, output_stride: int):
        if output_stride == 16:
            stage_list = [5]
            dilation_list = [2]
        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]
        else:
            raise ValueError(f"Output stride must be 16 or 8, got {output_stride}")

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_idx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_idx],
                dilation_rate=dilation_rate
            )
