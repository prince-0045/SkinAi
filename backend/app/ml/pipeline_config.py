"""
Pipeline-wide configuration for hair removal and classification.
All hyperparameters centralized — no magic numbers elsewhere.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ResolutionConfig:
    """Resolution policy: preserve detail for thin structures."""
    min_detection_size: int = 512
    classifier_size: int = 384
    interpolation: str = "bilinear"


@dataclass
class ClassicalPriorConfig:
    """Stage 1: Multi-scale blackhat morphology."""
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    adaptive_block_size: int = 11
    adaptive_c: int = 4
    closing_kernel: int = 3
    closing_iterations: int = 1
    min_object_area: int = 30


@dataclass
class DetectorConfig:
    """Stage 2: Deep thin-structure detector."""
    in_channels: int = 3
    base_filters: int = 32
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    fpn_channels: int = 64
    dropout: float = 0.1


@dataclass
class DirectionalConfig:
    """Stage 3: Gabor filter bank."""
    num_angles: int = 12
    frequencies: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2])
    sigma: float = 3.0
    gamma: float = 0.5


@dataclass
class FrequencyConfig:
    """Stage 4: Frequency domain channels."""
    gaussian_ksize: int = 21
    gaussian_sigma: float = 3.0
    laplacian_ksize: int = 5
    use_frangi: bool = False
    frangi_scales: Tuple[int, ...] = (1, 2, 3)


@dataclass
class ClassifierConfig:
    """Stage 5: Hair-robust classifier."""
    total_in_channels: int = 7
    backbone: str = "efficientnet_b0"
    num_classes: int = 5
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class LossConfig:
    """Loss weights for segmentation training."""
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    edge_weight: float = 0.3
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 2


@dataclass
class SyntheticHairConfig:
    """Config for synthetic hair generation (pseudo-label strategy)."""
    min_hairs: int = 5
    max_hairs: int = 30
    min_thickness: int = 1
    max_thickness: int = 3
    min_length: int = 80
    max_length: int = 250
    curvature_range: Tuple[float, float] = (0.0, 0.015)
    color_range: Tuple[int, int] = (10, 80)


@dataclass
class PipelineConfig:
    """Master config aggregating all sub-configs."""
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    classical: ClassicalPriorConfig = field(default_factory=ClassicalPriorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    synthetic: SyntheticHairConfig = field(default_factory=SyntheticHairConfig)
