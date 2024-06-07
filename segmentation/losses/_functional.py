import math
import numpy as np

from typing import Optional

import torch
from torch.nn import functional as F

__all__ = [
    "to_tensor",
    "soft_jaccard_score",
    "soft_dice_score"
]


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def soft_jaccard_score(
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output * target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output * target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp(min=eps)
    return jaccard_score


def soft_dice_score(
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None
) -> torch.Tensor:
    # print(f"Output size: {output.size()}, Target size: {target.size()}")
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dim_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dim_score
