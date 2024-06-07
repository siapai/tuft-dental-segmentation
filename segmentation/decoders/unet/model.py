from json import encoder
from typing import Optional, Union, List

from segmentation.encoders import get_encoder
from .decoder import UnetDecoder
from segmentation.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead
)


class Unet(SegmentationModel):
    """Unet is a fully convolutional neural network for semantic segmentation
    Arguments:
        encoder_name: Name pf the classification model (backbone) to extrack feature of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]
            Each stage generate features two times smaller
        encoder_weight: Pretrained weights
        decoder_channels: List of integers which specify **inc_channels** parameters. Length of the list should be
            the same as **encoder_depth**
        decoder_use_batchnorm: if **True**, BatchNorm2d layer between Conv2D and Activation layers is used.
            If **inplace** InplaceABN will be used, allows to decrease memory consumption
        decoder_attention_type: Attention module used in decoder of the model, available options are: **None**
        in_channels: A number of input channels for the model, default is 3 (RGB)
        classes: A number of classes for output mask.
        activation: An activation function to apply after the final convolution layer.
            Available options are: **None**, **sigmoid**, **softmax**, **identity**
        aux_params: Dictionary with parameters for auxiliary output (classification head).
            Auxiliary output is build on top of encoder if **aux_params** is not **None**
            Supported params:
            - classes (int): A number of classes
            - pooling (str): One of 'avg', 'max'
            - dropout (float): Dropout factor in (0, 1)
            - activation (str): An activation function to apply "sigmoid/softmax"
                Could be **None** to return logit

    Returns:
        `torch.nn.Module`: Unet
    """

    def __init__(
            self,
            encoder_name: str,
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = f"u-{encoder_name}"
        self.initialize()
