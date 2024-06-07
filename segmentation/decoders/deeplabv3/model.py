from typing import Optional

from torch import nn

from segmentation.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead
)

from segmentation.encoders import get_encoder
from .decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder


class DeepLabV3(SegmentationModel):
    """DeepLabV3 implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"
    Arguments:
        encoder_name: Name pf the classification model (backbone) to extrack feature of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]
            Each stage generate features two times smaller
        encoder_weight: Pretrained weights
        decoder_channels: List of integers which specify **inc_channels** parameters. Length of the list should be
            the same as **encoder_depth**
        in_channels: A number of input channels for the model, default is 3 (RGB)
        classes: A number of classes for output mask.
        activation: An activation function to apply after the final convolution layer.
            Available options are: **None**, **sigmoid**, **softmax**, **identity**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters for auxiliary output (classification head).
            Auxiliary output is build on top of encoder if **aux_params** is not **None**
            Supported params:
            - classes (int): A number of classes
            - pooling (str): One of 'avg', 'max'
            - dropout (float): Dropout factor in (0, 1)
            - activation (str): An activation function to apply "sigmoid/softmax"
                Could be **None** to return logit

    Returns:
        `torch.nn.Module`: **DeepLabV3**
    """
    def __init__(
            self,
            encoder_name: str = 'resnet34',
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = 'imagenet',
            decoder_channels: int = 256,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=8
        )

        self.decoder = DeepLabV3Decoder(
            in_channels=self.encoder.out_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = f"deeplabv3-{encoder_name}"
        self.initialize()


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

       Arguments:
           encoder_name: Name pf the classification model (backbone) to extrack feature of different spatial resolution
           encoder_depth: A number of stages used in encoder in range [3, 5]
               Each stage generate features two times smaller
           encoder_weight: Pretrained weights
           encoder_output_stride: Down sampling factor for encoder features
           decoder_atrous_rate: Dilation rates fo ASPP module
           decoder_channels: A number convolution filters in ASPP module. Default is 256
           in_channels: A number of input channels for the model, default is 3 (RGB)
           classes: A number of classes for output mask.
           activation: An activation function to apply after the final convolution layer.
               Available options are: **None**, **sigmoid**, **softmax**, **identity**
           upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
           aux_params: Dictionary with parameters for auxiliary output (classification head).
               Auxiliary output is build on top of encoder if **aux_params** is not **None**
               Supported params:
               - classes (int): A number of classes
               - pooling (str): One of 'avg', 'max'
               - dropout (float): Dropout factor in (0, 1)
               - activation (str): An activation function to apply "sigmoid/softmax"
                   Could be **None** to return logit

       Returns:
           `torch.nn.Module`: **DeepLabV3Plus**
       """

    def __init__(
            self,
            encoder_name: str = 'resnet34',
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = 'imagenet',
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        if encoder_output_stride not in {8, 16}:
            raise ValueError(f"Encoder output stride should be 8 or 16, got {encoder_output_stride}")

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride
        )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = f"deeplabv3plus-{encoder_name}"
        self.initialize()

