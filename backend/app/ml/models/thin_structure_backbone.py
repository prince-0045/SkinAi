"""
Stage 2a: Thin-Structure Backbone.
Encoder designed to preserve spatial resolution for thin hair strands.
Uses dilated convolutions instead of pooling for first two levels.
"""
import torch
import torch.nn as nn
from app.ml.pipeline_config import DetectorConfig


class DilatedConvBlock(nn.Module):
    """Conv → BN → ReLU with configurable dilation."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample via stride-2 conv, then a dilated conv."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.refine = DilatedConvBlock(out_ch, out_ch, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down(x))


class ThinStructureBackbone(nn.Module):
    """
    4-level encoder. First two levels use dilated convs (no pooling)
    to preserve thin-structure spatial detail.
    Returns feature maps at 4 scales.
    """

    def __init__(self, cfg: DetectorConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = DetectorConfig()
        f = cfg.base_filters
        dilations = cfg.dilations + [1]  # pad to 4 levels

        # Level 0: full resolution, dilated (no downsampling)
        self.level0 = nn.Sequential(
            DilatedConvBlock(cfg.in_channels, f, dilations[0]),
            DilatedConvBlock(f, f, dilations[0]),
        )
        # Level 1: half resolution, dilated
        self.level1 = DownBlock(f, f * 2, dilations[1])
        # Level 2: quarter resolution, dilated
        self.level2 = DownBlock(f * 2, f * 4, dilations[2])
        # Level 3: eighth resolution, standard
        self.level3 = DownBlock(f * 4, f * 8, dilations[3])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c0 = self.level0(x)
        c1 = self.level1(c0)
        c2 = self.level2(c1)
        c3 = self.level3(c2)
        return [c0, c1, c2, c3]
