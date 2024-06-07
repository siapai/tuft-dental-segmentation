from typing import Optional, List

import torch
from torch.nn import functional as F
from torch import nn
from ._functional import soft_jaccard_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["JaccardLoss"]


# noinspection PyProtectedMember
class JaccardLoss(nn.modules.loss._Loss):
    def __init__(
            self,
            mode: str,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = False,
            smooth: float = 0.0,
            eps: float = 1e-7
    ):
        """Jaccard loss (Iou) for image segmentation tasks
        Args:
            mode: Loss mode 'binary', 'multiclass', 'multilabel'
            classes: List of class that contribute in loss computation.
            log_loss: If True loss computed as `-log(jaccard_coeff)`, otherwise `1-jaccard_coeff`
            from_logits: if True, assumes input is raw logits
            smooth: Smoothness constants for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
        Shape
        - **y_pred** - torch.Tensor of shape (N, C, H, W)
        - **u_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        """

        assert mode in {BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE}
        super(JaccardLoss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking class is not supported with binary mode"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()


