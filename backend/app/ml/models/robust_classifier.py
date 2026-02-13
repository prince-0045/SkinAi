"""
Stage 5b: Hair-Robust Classifier.
Modified classifier head accepting 7-channel input.
Uses a learnable 1x1 projection to map 7→3 channels,
then feeds into a pretrained EfficientNet backbone.
The projection layer learns to suppress hair-affected pixels.
"""
import torch
import torch.nn as nn
from app.ml.pipeline_config import ClassifierConfig


class ChannelProjection(nn.Module):
    """Learnable 1x1 conv to project N channels → 3 for pretrained backbone."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 3, 1, bias=False),
            nn.BatchNorm2d(3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


def _build_backbone(cfg: ClassifierConfig) -> tuple[nn.Module, int]:
    """Build pretrained backbone, return (model_without_classifier, num_features)."""
    try:
        import timm
        model = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0)
        num_features = model.num_features
        return model, num_features
    except ImportError:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        model = efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, num_features


class RobustClassifier(nn.Module):
    """
    Hair-aware skin disease classifier.
    Input:  (B, 7, H, W) composed tensor
    Output: (B, num_classes) logits
    """

    def __init__(self, cfg: ClassifierConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ClassifierConfig()

        self.projection = ChannelProjection(cfg.total_in_channels)
        self.backbone, num_features = _build_backbone(cfg)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(num_features, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, 7, H, W) → (B, 3, H, W)
        features = self.backbone(x)  # (B, num_features)
        return self.head(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
