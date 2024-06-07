from typing import Optional, Union

from segmentation.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead
)

from segmentation.encoders import get_encoder
from .decoder import FPNDecoder


class FPN(SegmentationModel):
    """FPN is a fully convolutional neural network for image segmentation
    Arguments:
        encoder_name: Name of the classification model that will be used as backbone
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller than previous one. For example depth 0 we will have features with shapes [(N, C, H, W)]
            for depth 1 [(N, C, H, W), (N, C, H//2, W//2] and soon.
            Default is 5
        encoder_weights: One of **None**, **imagenet**.
        decoder_pyramid_channels: A number of convolutional filters in Feature Pyramid of FPN
        decoder_merge_policy: Determines how to merge pyramid features inside FPN. Available options **add**, **cat**
        decoder_dropout: Dropout rate in range (0, 1)
        in_channels: A number of input channels for the model, default is 3
        classes: A number of classes for output mask
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
        ``torch.nn.Module``: FPN
    """

    def __init__(
            self,
            encoder_name: str = 'resnet34',
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = 'imagenet',
            decoder_pyramid_channels: int = 256,
            decoder_segmentation_channels: int = 128,
            decoder_merge_policy: str = 'add',
            decoder_dropout: float = 0.2,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy
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

        self.name = f"fpn-{encoder_name}"
        self.initialize()

