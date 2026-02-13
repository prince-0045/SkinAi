"""
Stage 2: Hair Detector — top-level module.
Composes ThinStructureBackbone + FPNDecoder.
Thin wiring module — actual logic lives in sub-modules.
"""
import torch
import torch.nn as nn
from app.ml.pipeline_config import DetectorConfig
from app.ml.models.thin_structure_backbone import ThinStructureBackbone
from app.ml.models.fpn_decoder import FPNDecoder


class HairDetector(nn.Module):
    """
    Deep thin-structure hair detector.

    Input:  (B, 3, H, W) RGB image tensor
    Output: (mask_logits, confidence_map)
        mask_logits:     (B, 1, H, W) raw logits (apply sigmoid for prob)
        confidence_map:  (B, 1, H, W) pixel-wise detection confidence [0, 1]
    """

    def __init__(self, cfg: DetectorConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = DetectorConfig()
        self.backbone = ThinStructureBackbone(cfg)
        self.decoder = FPNDecoder(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mask_logits, confidence = self.decoder(features)
        return mask_logits, confidence

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convenience: returns binary mask (B, 1, H, W) uint8."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).to(torch.uint8)

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
