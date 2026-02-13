"""
Data augmentation + synthetic hair generation.
Creates paired (image, mask) training data by drawing realistic
curved hair lines on clean/low-hair dermoscopic images.
This is the key to training without ground-truth masks.
"""
import cv2
import numpy as np
from app.ml.pipeline_config import SyntheticHairConfig


def _random_bezier_curve(
    h: int, w: int, length: int, curvature: float
) -> list[tuple[int, int]]:
    """Generate a smooth random curve using quadratic Bezier."""
    x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
    angle = np.random.uniform(0, 2 * np.pi)
    x2 = int(x0 + length * np.cos(angle))
    y2 = int(y0 + length * np.sin(angle))

    # Control point with curvature offset
    mx, my = (x0 + x2) / 2, (y0 + y2) / 2
    offset = length * curvature * np.random.choice([-1, 1])
    cx = int(mx + offset * np.sin(angle))
    cy = int(my - offset * np.cos(angle))

    points = []
    for t in np.linspace(0, 1, max(length, 50)):
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x2
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y2
        points.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    return points


def generate_synthetic_hairs(
    image: np.ndarray, cfg: SyntheticHairConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw realistic hair-like curves on an image.
    Returns: (hairy_image, hair_mask)  both (H,W,3)/(H,W) uint8
    """
    if cfg is None:
        cfg = SyntheticHairConfig()

    h, w = image.shape[:2]
    hairy = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    n_hairs = np.random.randint(cfg.min_hairs, cfg.max_hairs + 1)

    for _ in range(n_hairs):
        thickness = np.random.randint(cfg.min_thickness, cfg.max_thickness + 1)
        length = np.random.randint(cfg.min_length, cfg.max_length + 1)
        curvature = np.random.uniform(*cfg.curvature_range)
        color_val = np.random.randint(*cfg.color_range)
        color = (color_val, color_val, color_val)

        points = _random_bezier_curve(h, w, length, curvature)
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(hairy, [pts], False, color, thickness, cv2.LINE_AA)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)

    return hairy, mask


def augment_spatial(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Basic spatial augmentations: flip + rotation."""
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if np.random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    angle = np.random.choice([0, 90, 180, 270])
    if angle > 0:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
    return image, mask


def augment_photometric(image: np.ndarray) -> np.ndarray:
    """Color jitter: brightness, contrast, saturation adjustments."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)  # saturation
    hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)  # brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Contrast
    alpha = np.random.uniform(0.8, 1.2)
    result = np.clip(alpha * result.astype(np.float32), 0, 255).astype(np.uint8)
    return result
