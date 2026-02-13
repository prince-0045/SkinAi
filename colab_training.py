#!/usr/bin/env python3
"""
=================================================================
  SkinAI — Complete Hair Removal Pipeline Training Script (Colab)
=================================================================
Self-contained. All model definitions, training loops, evaluation,
and data generation are inlined here. No external project imports.

Usage on Google Colab:
  1. Upload this file or paste cells into a notebook
  2. Upload your dermoscopic images to /content/images/
  3. Run all cells — models will be saved to /content/checkpoints/
  4. Download checkpoints and place in your backend's checkpoints/ dir
=================================================================
"""
# ──────────────────────────────────────────────────────
# CELL 1: Install Dependencies
# ──────────────────────────────────────────────────────
import subprocess, sys

def install_deps():
    """Install required packages on Colab."""
    pkgs = ["torch", "torchvision", "timm", "scikit-image", "albumentations", "opencv-python-headless"]
    for pkg in pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All dependencies installed.")

# Uncomment the next line on Colab:
install_deps()

# ──────────────────────────────────────────────────────
# CELL 2: Imports
# ──────────────────────────────────────────────────────
import os, glob, time, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print(f"🔧 PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────
# CELL 3: Configuration (all hyperparameters in one place)
# ──────────────────────────────────────────────────────
@dataclass
class ResolutionConfig:
    min_detection_size: int = 512
    classifier_size: int = 384
    interpolation: str = "bilinear"

@dataclass
class ClassicalPriorConfig:
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    adaptive_block_size: int = 11
    adaptive_c: int = 4
    closing_kernel: int = 3
    closing_iterations: int = 1
    min_object_area: int = 30

@dataclass
class DetectorConfig:
    in_channels: int = 3
    base_filters: int = 32
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    fpn_channels: int = 64
    dropout: float = 0.1

@dataclass
class DirectionalConfig:
    num_angles: int = 12
    frequencies: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2])
    sigma: float = 3.0
    gamma: float = 0.5

@dataclass
class FrequencyConfig:
    gaussian_ksize: int = 21
    gaussian_sigma: float = 3.0
    laplacian_ksize: int = 5
    use_frangi: bool = False
    frangi_scales: Tuple[int, ...] = (1, 2, 3)

@dataclass
class ClassifierConfig:
    total_in_channels: int = 7
    backbone: str = "efficientnet_b0"
    num_classes: int = 5
    pretrained: bool = True
    dropout: float = 0.3

@dataclass
class LossConfig:
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    edge_weight: float = 0.3
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs_detector: int = 50
    epochs_classifier: int = 30
    warmup_epochs: int = 3
    mixed_precision: bool = True
    checkpoint_dir: str = "/content/checkpoints"
    num_workers: int = 2

@dataclass
class SyntheticHairConfig:
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
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    classical: ClassicalPriorConfig = field(default_factory=ClassicalPriorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    synthetic: SyntheticHairConfig = field(default_factory=SyntheticHairConfig)

CFG = PipelineConfig()
print("✅ Configuration loaded.")


# ==============================================================
# STAGE 1: Classical Hair Prior (pure OpenCV)
# ==============================================================
def _blackhat_at_scale(gray, ksize):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

def _adaptive_threshold(response, cfg):
    if response.max() < 15:
        return np.zeros_like(response)
    _, global_mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, cfg.adaptive_block_size, cfg.adaptive_c,
    )
    return cv2.bitwise_and(global_mask, adaptive)

def _morphological_close(mask, cfg):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.closing_kernel, cfg.closing_kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.closing_iterations)

def _remove_small_objects(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

def extract_hair_mask(image, config=None):
    """Multi-scale blackhat → adaptive threshold → morph close → clean."""
    if config is None:
        config = ClassicalPriorConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(gray)
    for ksize in config.kernel_sizes:
        response = _blackhat_at_scale(gray, ksize)
        binary = _adaptive_threshold(response, config)
        combined = cv2.bitwise_or(combined, binary)
    closed = _morphological_close(combined, config)
    return _remove_small_objects(closed, config.min_object_area)


# ==============================================================
# STAGE 2: Deep Thin-Structure Detector (PyTorch)
# ==============================================================
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.refine = DilatedConvBlock(out_ch, out_ch, dilation)
    def forward(self, x): return self.refine(self.down(x))

class ThinStructureBackbone(nn.Module):
    """4-level encoder with dilated convolutions (preserves thin structures)."""
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = DetectorConfig()
        f = cfg.base_filters
        d = cfg.dilations + [1]
        self.level0 = nn.Sequential(DilatedConvBlock(cfg.in_channels, f, d[0]), DilatedConvBlock(f, f, d[0]))
        self.level1 = DownBlock(f, f*2, d[1])
        self.level2 = DownBlock(f*2, f*4, d[2])
        self.level3 = DownBlock(f*4, f*8, d[3])
    def forward(self, x):
        c0 = self.level0(x); c1 = self.level1(c0)
        c2 = self.level2(c1); c3 = self.level3(c2)
        return [c0, c1, c2, c3]

class LateralBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x): return self.bn(self.conv(x))

class SmoothBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class FPNDecoder(nn.Module):
    """FPN that fuses 4-level features → mask logits + confidence."""
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = DetectorConfig()
        f = cfg.base_filters; fpn = cfg.fpn_channels
        enc_ch = [f, f*2, f*4, f*8]
        self.laterals = nn.ModuleList([LateralBlock(c, fpn) for c in enc_ch])
        self.smooths = nn.ModuleList([SmoothBlock(fpn) for _ in enc_ch])
        self.merge = nn.Conv2d(fpn*4, fpn, 1, bias=False)
        self.mask_head = nn.Conv2d(fpn, 1, 1)
        self.conf_head = nn.Conv2d(fpn, 1, 1)
        self.dropout = nn.Dropout2d(cfg.dropout)

    def forward(self, features):
        target = features[0].shape[2:]
        lats = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(lats)-1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i-1].shape[2:], mode="bilinear", align_corners=False)
            lats[i-1] = lats[i-1] + up
        smoothed = [s(l) for s, l in zip(self.smooths, lats)]
        ups = [F.interpolate(s, size=target, mode="bilinear", align_corners=False) for s in smoothed]
        merged = self.dropout(self.merge(torch.cat(ups, dim=1)))
        return self.mask_head(merged), torch.sigmoid(self.conf_head(merged))

class HairDetector(nn.Module):
    """ThinStructureBackbone + FPNDecoder = Hair Detector."""
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = DetectorConfig()
        self.backbone = ThinStructureBackbone(cfg)
        self.decoder = FPNDecoder(cfg)
    def forward(self, x):
        return self.decoder(self.backbone(x))
    def predict_mask(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return (torch.sigmoid(logits) > threshold).to(torch.uint8)

n_params = sum(p.numel() for p in HairDetector().parameters() if p.requires_grad)
print(f"✅ HairDetector: {n_params:,} trainable parameters")


# ==============================================================
# STAGE 3: Directional Enhancement (Gabor filter bank)
# ==============================================================
def extract_directional_map(image, config=None):
    if config is None: config = DirectionalConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    angles = np.linspace(0, np.pi, config.num_angles, endpoint=False)
    responses = []
    for theta in angles:
        for freq in config.frequencies:
            k = cv2.getGaborKernel((21,21), config.sigma, theta, 1.0/freq, config.gamma, 0, cv2.CV_32F)
            k /= k.sum() + 1e-7
            responses.append(cv2.filter2D(gray, cv2.CV_32F, k))
    response = np.max(np.stack(responses, axis=0), axis=0)
    rmin, rmax = response.min(), response.max()
    if rmax - rmin < 1e-6: return np.zeros_like(gray, dtype=np.uint8)
    return ((response - rmin) / (rmax - rmin) * 255.0).astype(np.uint8)


# ==============================================================
# STAGE 4: Frequency Domain Channels
# ==============================================================
def extract_frequency_channels(image, config=None):
    if config is None: config = FrequencyConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # High-pass
    blurred = cv2.GaussianBlur(gray, (config.gaussian_ksize, config.gaussian_ksize), config.gaussian_sigma)
    ch0 = cv2.subtract(gray, blurred)
    # Laplacian
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=config.laplacian_ksize))
    lmax = lap.max()
    ch1 = (lap / lmax * 255.0).astype(np.uint8) if lmax > 0 else np.zeros_like(gray)
    # Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mmax = mag.max()
    ch2 = (mag / mmax * 255.0).astype(np.uint8) if mmax > 0 else np.zeros_like(gray)
    return np.stack([ch0, ch1, ch2], axis=-1)


# ==============================================================
# STAGE 5a: Channel Composer
# ==============================================================
def compose_channels(rgb, hair_mask, directional, freq_ch):
    """Compose 7-channel input: RGB(3) + Mask(1) + Dir(1) + Freq(2)."""
    rgb_f = rgb.astype(np.float32) / 255.0
    mask_f = hair_mask.astype(np.float32) / 255.0
    dir_f = directional.astype(np.float32) / 255.0
    freq_f = freq_ch[..., :2].astype(np.float32) / 255.0
    return np.concatenate([rgb_f, mask_f[..., None], dir_f[..., None], freq_f], axis=-1)


# ==============================================================
# STAGE 5b: Robust Classifier
# ==============================================================
class ChannelProjection(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 3, 1, bias=False), nn.BatchNorm2d(3),
        )
    def forward(self, x): return self.project(x)

def _build_backbone(cfg):
    try:
        import timm
        model = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=0)
        return model, model.num_features
    except ImportError:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        w = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        model = efficientnet_b0(weights=w)
        nf = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, nf

class RobustClassifier(nn.Module):
    """7-channel → projection → EfficientNet → class logits."""
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = ClassifierConfig()
        self.projection = ChannelProjection(cfg.total_in_channels)
        self.backbone, nf = _build_backbone(cfg)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(nf, cfg.num_classes))
    def forward(self, x):
        return self.head(self.backbone(self.projection(x)))


# ==============================================================
# LOSS FUNCTIONS
# ==============================================================
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        p = torch.sigmoid(pred).flatten(1); t = target.flatten(1)
        inter = (p * t).sum(1); union = p.sum(1) + t.sum(1)
        return 1.0 - ((2*inter + 1) / (union + 1)).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = target * torch.sigmoid(pred) + (1-target) * (1-torch.sigmoid(pred))
        at = target * self.alpha + (1-target) * (1-self.alpha)
        return (at * ((1-pt)**self.gamma) * bce).mean()

class EdgeConsistencyLoss(nn.Module):
    def forward(self, pred, target):
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=pred.dtype, device=pred.device).view(1,1,3,3)
        sy = sx.transpose(2,3)
        edges = torch.sqrt(F.conv2d(target,sx,padding=1)**2 + F.conv2d(target,sy,padding=1)**2 + 1e-8)
        edges = edges / (edges.max() + 1e-8)
        return (torch.abs(torch.sigmoid(pred) - target) * edges).mean()

class CombinedSegmentationLoss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None: cfg = LossConfig()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(cfg.focal_alpha, cfg.focal_gamma)
        self.edge = EdgeConsistencyLoss()
    def forward(self, pred, target):
        lb = self.bce(pred, target); ld = self.dice(pred, target)
        lf = self.focal(pred, target); le = self.edge(pred, target)
        total = self.cfg.bce_weight*lb + self.cfg.dice_weight*ld + self.cfg.focal_weight*lf + self.cfg.edge_weight*le
        return {"total": total, "bce": lb.item(), "dice": ld.item(), "focal": lf.item(), "edge": le.item()}

print("✅ Loss functions ready.")


# ==============================================================
# SYNTHETIC HAIR GENERATION (training without ground-truth masks)
# ==============================================================
def _random_bezier_curve(h, w, length, curvature):
    x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
    angle = np.random.uniform(0, 2*np.pi)
    x2, y2 = int(x0 + length*np.cos(angle)), int(y0 + length*np.sin(angle))
    mx, my = (x0+x2)/2, (y0+y2)/2
    offset = length * curvature * np.random.choice([-1, 1])
    cx, cy = int(mx + offset*np.sin(angle)), int(my - offset*np.cos(angle))
    points = []
    for t in np.linspace(0, 1, max(length, 50)):
        x = (1-t)**2*x0 + 2*(1-t)*t*cx + t**2*x2
        y = (1-t)**2*y0 + 2*(1-t)*t*cy + t**2*y2
        points.append((int(np.clip(x,0,w-1)), int(np.clip(y,0,h-1))))
    return points

def generate_synthetic_hairs(image, cfg=None):
    """Draw realistic Bezier-curved dark lines → (hairy_image, hair_mask)."""
    if cfg is None: cfg = SyntheticHairConfig()
    h, w = image.shape[:2]
    hairy, mask = image.copy(), np.zeros((h,w), dtype=np.uint8)
    for _ in range(np.random.randint(cfg.min_hairs, cfg.max_hairs+1)):
        thick = np.random.randint(cfg.min_thickness, cfg.max_thickness+1)
        length = np.random.randint(cfg.min_length, cfg.max_length+1)
        curv = np.random.uniform(*cfg.curvature_range)
        cval = np.random.randint(*cfg.color_range)
        pts = np.array(_random_bezier_curve(h, w, length, curv), dtype=np.int32)
        cv2.polylines(hairy, [pts], False, (cval,cval,cval), thick, cv2.LINE_AA)
        cv2.polylines(mask, [pts], False, 255, thick, cv2.LINE_AA)
    return hairy, mask

def augment_spatial(image, mask):
    if np.random.random() > 0.5: image, mask = cv2.flip(image,1), cv2.flip(mask,1)
    if np.random.random() > 0.5: image, mask = cv2.flip(image,0), cv2.flip(mask,0)
    angle = np.random.choice([0, 90, 180, 270])
    if angle > 0:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image, mask = cv2.warpAffine(image, M, (w,h)), cv2.warpAffine(mask, M, (w,h))
    return image, mask

def augment_photometric(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= np.random.uniform(0.8, 1.2)
    hsv[:,:,2] *= np.random.uniform(0.8, 1.2)
    result = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(np.random.uniform(0.8,1.2) * result.astype(np.float32), 0, 255).astype(np.uint8)


# ==============================================================
# DATASET: Supports pseudo-label + synthetic hair modes
# ==============================================================
class HairDetectorDataset(Dataset):
    """
    Loads dermoscopic images and generates training pairs.
    Mode 1: Synthetic hair on CLEAN images → perfect paired data
    Mode 2: Pseudo-labels via classical prior → noisy but real data
    """
    def __init__(self, image_dir, size=512, mode="synthetic"):
        self.paths = sorted(glob.glob(os.path.join(image_dir, "**/*.*"), recursive=True))
        self.paths = [p for p in self.paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        self.size = size
        self.mode = mode
        print(f"📂 Dataset: {len(self.paths)} images, mode={mode}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.random.randint(100, 200, (self.size, self.size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size))

        if self.mode == "synthetic":
            img = augment_photometric(img)
            hairy, mask = generate_synthetic_hairs(img)
            hairy, mask = augment_spatial(hairy, mask)
            image_out = hairy
        else:  # pseudo-label
            mask = extract_hair_mask(img)
            img = augment_photometric(img)
            img, mask = augment_spatial(img, mask)
            image_out = img

        tensor_img = torch.from_numpy(image_out.transpose(2,0,1)).float() / 255.0
        tensor_mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return tensor_img, tensor_mask


# ==============================================================
# TRAINING: Hair Detector (Stage 2)
# ==============================================================
def train_hair_detector(image_dir, cfg=None):
    """
    Full training loop for the hair detector.
    Uses synthetic hair generation for training data.
    """
    if cfg is None: cfg = PipelineConfig()
    print("\n" + "="*60)
    print("  PHASE 1: Training Hair Detector")
    print("="*60)

    # Data
    ds = HairDetectorDataset(image_dir, cfg.resolution.min_detection_size, mode="synthetic")
    n_val = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val])
    train_dl = DataLoader(train_ds, cfg.training.batch_size, shuffle=True,
                          num_workers=cfg.training.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.num_workers, pin_memory=True)

    # Model
    model = HairDetector(cfg.detector).to(DEVICE)
    criterion = CombinedSegmentationLoss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs_detector)
    scaler = torch.amp.GradScaler("cuda") if cfg.training.mixed_precision and DEVICE.type == "cuda" else None

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg.training.epochs_detector):
        # ── Train ──
        model.train()
        train_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(imgs)
                losses = criterion(logits, masks)
            loss = losses["total"]
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits, _ = model(imgs)
                losses = criterion(logits, masks)
                val_loss += losses["total"].item()
                # IoU metric
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * masks).sum()
                union = pred.sum() + masks.sum() - inter
                val_dice += (2*inter / (pred.sum() + masks.sum() + 1e-8)).item()
        val_loss /= len(val_dl)
        val_dice /= len(val_dl)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:3d}/{cfg.training.epochs_detector} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Dice: {val_dice:.3f} | LR: {lr:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_hair_detector.pth")
            print(f"    💾 Saved best model (val_loss={best_loss:.4f})")

    torch.save(model.state_dict(), ckpt_dir / "final_hair_detector.pth")
    print(f"\n✅ Detector training complete. Best val loss: {best_loss:.4f}")
    print(f"   Checkpoints saved to: {ckpt_dir}")
    return model


# ==============================================================
# COMPOSE 7-CHANNEL TENSORS (prep for classifier training)
# ==============================================================
def compose_dataset_for_classifier(image_dir, detector_model, output_dir, cfg=None):
    """
    Run the full pipeline on all images and save 7-channel .npy tensors.
    This bridges detector training → classifier training.
    """
    if cfg is None: cfg = PipelineConfig()
    print("\n" + "="*60)
    print("  PHASE 2: Composing 7-Channel Tensors")
    print("="*60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(image_dir, "**/*.*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))]

    detector_model.eval()
    composed_files = []

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (cfg.resolution.min_detection_size, cfg.resolution.min_detection_size))

        # Stage 1: Classical hair mask
        classical_mask = extract_hair_mask(img, cfg.classical)

        # Stage 2: Deep detection
        tensor_in = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            deep_mask = detector_model.predict_mask(tensor_in)
            deep_mask = (deep_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Union
        hair_mask = cv2.bitwise_or(classical_mask, deep_mask)

        # Stage 3: Directional
        directional = extract_directional_map(img, cfg.directional)

        # Stage 4: Frequency
        freq = extract_frequency_channels(img, cfg.frequency)

        # Stage 5a: Compose
        composed = compose_channels(img, hair_mask, directional, freq)

        # Resize for classifier
        cls_size = cfg.resolution.classifier_size
        composed = cv2.resize(composed, (cls_size, cls_size))

        fname = f"{i:05d}.npy"
        np.save(out / fname, composed)
        composed_files.append((fname, Path(path).stem))

        if (i+1) % 50 == 0 or i == len(paths)-1:
            print(f"  Composed {i+1}/{len(paths)} images")

    print(f"✅ Saved {len(composed_files)} composed tensors to {out}")
    return composed_files


# ==============================================================
# CLASSIFIER DATASET
# ==============================================================
class ClassifierDataset(Dataset):
    """Loads pre-composed 7-channel .npy files + labels."""
    def __init__(self, data_dir, label_file, size=384):
        self.data_dir = Path(data_dir)
        self.size = size
        self.samples = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))
        print(f"📂 Classifier dataset: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        data = np.load(self.data_dir / fname)
        if data.shape[0] != self.size or data.shape[1] != self.size:
            data = cv2.resize(data, (self.size, self.size))
        return torch.from_numpy(data.transpose(2,0,1)).float(), label


# ==============================================================
# TRAINING: Robust Classifier (Stage 5)
# ==============================================================
def train_classifier(data_dir, label_file, cfg=None):
    """Train the 7-channel hair-robust classifier."""
    if cfg is None: cfg = PipelineConfig()
    print("\n" + "="*60)
    print("  PHASE 3: Training Hair-Robust Classifier")
    print("="*60)

    ds = ClassifierDataset(data_dir, label_file, cfg.resolution.classifier_size)
    n_val = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds)-n_val, n_val])
    train_dl = DataLoader(train_ds, cfg.training.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, cfg.training.batch_size)

    model = RobustClassifier(cfg.classifier).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), cfg.training.learning_rate,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs_classifier)

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.training.epochs_classifier):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_acc = correct / max(total, 1)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(f"  Epoch {epoch+1:3d}/{cfg.training.epochs_classifier} | "
              f"Loss: {total_loss/len(train_dl):.4f} | Acc: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best_classifier.pth")
            print(f"    💾 Saved best classifier (val_acc={best_acc:.3f})")

    print(f"\n✅ Classifier training complete. Best val acc: {best_acc:.3f}")
    return model


# ==============================================================
# EVALUATION
# ==============================================================
def evaluate_detector(model, image_dir, cfg=None, num_samples=50):
    """Quick evaluation of detector on a subset of images."""
    if cfg is None: cfg = PipelineConfig()
    print("\n" + "="*60)
    print("  EVALUATION: Hair Detector")
    print("="*60)

    paths = sorted(glob.glob(os.path.join(image_dir, "**/*.*"), recursive=True))
    paths = [p for p in paths if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))][:num_samples]

    model.eval()
    total_recall, total_dice, total_border = 0, 0, 0
    n = 0

    for path in paths:
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (cfg.resolution.min_detection_size, cfg.resolution.min_detection_size))

        # Generate synthetic hair for evaluation (since we have no GT masks)
        hairy, gt_mask = generate_synthetic_hairs(img.copy())
        tensor = torch.from_numpy(hairy.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            pred_mask = model.predict_mask(tensor).squeeze().cpu().numpy() * 255

        gt_b = (gt_mask > 127).astype(bool)
        pred_b = (pred_mask > 127).astype(bool)
        tp = (pred_b & gt_b).sum()
        recall = tp / max(gt_b.sum(), 1)
        dice = 2*tp / max(pred_b.sum() + gt_b.sum(), 1)
        total_recall += recall
        total_dice += dice
        n += 1

    if n > 0:
        print(f"  Samples evaluated: {n}")
        print(f"  Mean Recall:       {total_recall/n:.3f}  (target: >0.80)")
        print(f"  Mean Dice:         {total_dice/n:.3f}  (target: >0.60)")
    else:
        print("  ⚠️ No valid images found for evaluation.")


# ==============================================================
# 🚀 MAIN EXECUTION — Run everything!
# ==============================================================
if __name__ == "__main__":
    # ─── Configuration ───
    IMAGE_DIR = "/content/images"          # ← PUT YOUR DERMOSCOPIC IMAGES HERE
    COMPOSED_DIR = "/content/composed"     # Auto-generated
    LABEL_FILE = "/content/labels.txt"     # ← CREATE THIS (see instructions below)

    print("\n" + "🔬"*30)
    print("  SkinAI Hair Removal Pipeline — Training")
    print("🔬"*30)

    # ──────────────────────────────────────────────────
    # STEP 1: Upload your images
    # ──────────────────────────────────────────────────
    if not os.path.exists(IMAGE_DIR):
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  📁 SETUP REQUIRED                                      ║
║                                                          ║
║  1. Create folder: {IMAGE_DIR:<38s} ║
║  2. Upload your dermoscopic images (.jpg/.png) there     ║
║  3. Re-run this script                                   ║
║                                                          ║
║  On Colab, run:                                          ║
║    !mkdir -p /content/images                             ║
║    # Then use the file upload widget or:                 ║
║    # from google.colab import files                      ║
║    # files.upload()                                      ║
╚══════════════════════════════════════════════════════════╝
""")
    else:
        n_imgs = len([f for f in glob.glob(os.path.join(IMAGE_DIR,"**/*.*"), recursive=True)
                      if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
        print(f"\n📂 Found {n_imgs} images in {IMAGE_DIR}")

        if n_imgs > 0:
            # ──────────────────────────────────────────
            # STEP 2: Train hair detector
            # ──────────────────────────────────────────
            detector = train_hair_detector(IMAGE_DIR, CFG)

            # ──────────────────────────────────────────
            # STEP 3: Evaluate detector
            # ──────────────────────────────────────────
            evaluate_detector(detector, IMAGE_DIR, CFG)

            # ──────────────────────────────────────────
            # STEP 4: Compose 7-channel tensors
            # ──────────────────────────────────────────
            composed_files = compose_dataset_for_classifier(IMAGE_DIR, detector, COMPOSED_DIR, CFG)

            # ──────────────────────────────────────────
            # STEP 5: Train classifier (if labels exist)
            # ──────────────────────────────────────────
            if os.path.exists(LABEL_FILE):
                classifier = train_classifier(COMPOSED_DIR, LABEL_FILE, CFG)
                print("\n✅ FULL PIPELINE TRAINING COMPLETE!")
            else:
                print(f"""
╔══════════════════════════════════════════════════════════╗
║  📝 LABEL FILE REQUIRED FOR CLASSIFIER                   ║
║                                                          ║
║  Create: {LABEL_FILE:<47s} ║
║  Format: one line per image                              ║
║                                                          ║
║    00000.npy 0                                           ║
║    00001.npy 2                                           ║
║    00002.npy 1                                           ║
║                                                          ║
║  Labels: 0=Eczema 1=Psoriasis 2=Melanoma                ║
║          3=Acne   4=Benign Keratosis                     ║
║                                                          ║
║  Then re-run to train the classifier.                    ║
╚══════════════════════════════════════════════════════════╝
""")

            # ──────────────────────────────────────────
            # STEP 6: Download checkpoints
            # ──────────────────────────────────────────
            print(f"""
╔══════════════════════════════════════════════════════════╗
║  📦 DEPLOYMENT                                           ║
║                                                          ║
║  Download these files from {CFG.training.checkpoint_dir}:║
║    • best_hair_detector.pth                              ║
║    • best_classifier.pth  (if classifier was trained)    ║
║                                                          ║
║  Place them in your backend's checkpoints/ directory:    ║
║    backend/checkpoints/best_hair_detector.pth            ║
║    backend/checkpoints/best_classifier.pth               ║
║                                                          ║
║  The API will auto-detect and load them!                 ║
╚══════════════════════════════════════════════════════════╝
""")
        else:
            print("⚠️ No images found. Please upload images first.")
