"""
Stage 2b: Feature Pyramid Network Decoder.
Fuses multi-scale features from ThinStructureBackbone
via lateral connections and top-down pathway.
Outputs hair mask logits and confidence map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from app.ml.pipeline_config import DetectorConfig


class LateralBlock(nn.Module):
    """1x1 conv to project encoder features to FPN channel dim."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class SmoothBlock(nn.Module):
    """3x3 conv to reduce aliasing after upsampling + addition."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPNDecoder(nn.Module):
    """
    Top-down FPN that fuses 4-level encoder features.
    Produces single-channel mask logits + confidence map.
    """

    def __init__(self, cfg: DetectorConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = DetectorConfig()
        f = cfg.base_filters
        fpn_ch = cfg.fpn_channels
        encoder_channels = [f, f * 2, f * 4, f * 8]

        self.laterals = nn.ModuleList(
            [LateralBlock(ch, fpn_ch) for ch in encoder_channels]
        )
        self.smooths = nn.ModuleList(
            [SmoothBlock(fpn_ch) for _ in encoder_channels]
        )
        # Final head: merge all levels → mask + confidence
        self.merge = nn.Conv2d(fpn_ch * 4, fpn_ch, 1, bias=False)
        self.mask_head = nn.Conv2d(fpn_ch, 1, 1)
        self.conf_head = nn.Conv2d(fpn_ch, 1, 1)
        self.dropout = nn.Dropout2d(cfg.dropout)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = features[0].shape[2:]
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        smoothed = [s(l) for s, l in zip(self.smooths, laterals)]
        # Upsample all to full resolution and concat
        upsampled = [F.interpolate(s, size=target_size, mode="bilinear", align_corners=False) for s in smoothed]
        merged = self.merge(torch.cat(upsampled, dim=1))
        merged = self.dropout(merged)

        mask_logits = self.mask_head(merged)
        confidence = torch.sigmoid(self.conf_head(merged))
        return mask_logits, confidence
