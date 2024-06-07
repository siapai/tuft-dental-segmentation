from . import encoders
from . import decoders
from . import losses

from .decoders.unet import Unet
from .decoders.fpn import FPN
from .decoders.deeplabv3 import DeepLabV3, DeepLabV3Plus
from .decoders.unetplusplus import UnetPlusPlus
from .decoders.pspnet import PSPNet
from .decoders.pan import PAN

from .__version__ import __version__

from typing import Optional as _Optional
import torch as _torch


def create_model(
        arch: str,
        encoder_name: str = "resnet34",
        encoder_weights: _Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        **kwargs
) -> _torch.nn.Module:
    """Module entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    archs = [
        Unet,
        FPN,
        DeepLabV3,
        DeepLabV3Plus,
        UnetPlusPlus,
        PSPNet,
        PAN
    ]

    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(f"Wrong architecture type {arch}. Available options are: {list(archs_dict.keys())}")
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs
    )
