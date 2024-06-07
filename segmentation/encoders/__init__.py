import functools
from torch.utils import model_zoo

from ._preprocessing import preprocess_input

from .restnet import resnet_encoders
from .mobilenet import mobilenet_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(mobilenet_encoders)


def get_encoder(name: str, in_channels: int = 3, depth: int = 5, weights=None, output_stride: int = 32):
    try:
        base_encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(f"Wrong encoder name {name}, supported encoders: {list(encoders.keys())}")

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = base_encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(f"Wrong pretrained weights {weights}, for encoder {name}."
                           f"\nAvailable options are: {list(encoders[name])}")
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    all_settings = encoders[encoder_name]["pretrained_settings"]
    if pretrained not in all_settings.keys():
        raise ValueError(f"Available pretrained options {all_settings.keys()}")
    settings = all_settings[pretrained]

    formatted_settings = {
        "input_shape": settings.get("input_shape", "RGB"),
        "input_range": list(settings.get("input_range", [0, 1])),
        "mean": list(settings.get("mean")),
        "std": list(settings.get("std"))
    }

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)


