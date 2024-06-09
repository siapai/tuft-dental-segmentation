from typing import Optional, Union

from segmentation.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead
)

from segmentation.encoders import get_encoder
from .decoder import LinknetDecoder


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
       and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
       resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *sum*
       for fusing decoder blocks with skip connections.
   """
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            decoder_use_batchnorm: bool = True,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights
        )

        self.decoder = LinknetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_batchnorm=decoder_use_batchnorm
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32, out_channels=classes, activation=activation, kernel_size=1
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = f"link-{encoder_name}"
        self.initialize()
