"""
Loss engineering for hair segmentation training.
Combined loss: α*BCE + β*Dice + γ*Focal + δ*EdgeConsistency
Handles severe class imbalance (<3% hair pixels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.ml.pipeline_config import LossConfig


class DiceLoss(nn.Module):
    """Dice loss for class-imbalanced binary segmentation."""
    smooth: float = 1.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sig = torch.sigmoid(pred).flatten(1)
        target_f = target.flatten(1)
        intersection = (pred_sig * target_f).sum(1)
        union = pred_sig.sum(1) + target_f.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy negatives for sparse hair pixels."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pred_sig = torch.sigmoid(pred)
        pt = target * pred_sig + (1 - target) * (1 - pred_sig)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class EdgeConsistencyLoss(nn.Module):
    """Penalizes mask predictions that corrupt lesion edge regions."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Sobel edge detection on target to find lesion borders
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)

        edge_x = F.conv2d(target, sobel_x, padding=1)
        edge_y = F.conv2d(target, sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        edges = edges / (edges.max() + 1e-8)  # normalize

        # Penalize prediction errors near edges
        pred_sig = torch.sigmoid(pred)
        error = torch.abs(pred_sig - target)
        edge_error = error * edges
        return edge_error.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    L_total = α*BCE + β*Dice + γ*Focal + δ*EdgeConsistency
    """

    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = LossConfig()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.edge = EdgeConsistencyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l_bce = self.bce(pred, target)
        l_dice = self.dice(pred, target)
        l_focal = self.focal(pred, target)
        l_edge = self.edge(pred, target)

        total = (
            self.cfg.bce_weight * l_bce
            + self.cfg.dice_weight * l_dice
            + self.cfg.focal_weight * l_focal
            + self.cfg.edge_weight * l_edge
        )
        return {
            "total": total, "bce": l_bce.item(), "dice": l_dice.item(),
            "focal": l_focal.item(), "edge": l_edge.item(),
        }
