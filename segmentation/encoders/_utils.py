import torch
from torch import nn


def patch_first_conv(
        model,
        new_in_channels: int,
        default_in_channels: int = 3,
        pretrained: bool = True
):
    """Change the first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization

    Args:
        model (EncoderMixin): The model whose first convolution layer needs to be patched.
        new_in_channels (int): The number of new input channels.
        default_in_channels (int): The number of original input channels (default is 3 for RGB images).
        pretrained (bool): Whether the model is pretrained or not.
    """

    first_conv = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            first_conv = module
            break

    weight = first_conv.weight.detach()
    first_conv.in_channels = new_in_channels

    if not pretrained:
        # Initialize new weights
        first_conv.weight = nn.Parameter(
            torch.Tensor(first_conv.out_channels, new_in_channels // first_conv.groups, *first_conv.kernel_size)
        )
        first_conv.reset_parameters()

    elif new_in_channels == 1:
        # Sum the weights along the channel dimension
        new_weight = weight.sum(1, keepdim=True)
        first_conv.weight = nn.Parameter(new_weight)

    else:
        # Initialize new weights and handle the case where new_in_channels > 3
        new_weight = torch.Tensor(
            first_conv.out_channels,
            new_in_channels // first_conv.groups,
            *first_conv.kernel_size
        )
        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        first_conv.weight = nn.Parameter(new_weight)


def replace_strides_with_dilation(module: nn.Module, dilation_rate: int):
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # For EfficientNet
            if hasattr(mod, 'static_padding'):
                mod.static_padding = nn.Identity()
