"""
Stage 1: Multi-Scale Classical Hair Prior.
Uses morphological blackhat at multiple kernel sizes to detect
dark thin structures on lighter skin backgrounds.
No deep learning — pure OpenCV.
"""
import cv2
import numpy as np
from app.ml.pipeline_config import ClassicalPriorConfig


def _blackhat_at_scale(gray: np.ndarray, ksize: int) -> np.ndarray:
    """Apply blackhat transform with a cross-shaped kernel."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (ksize, ksize)
    )
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def _adaptive_threshold(response: np.ndarray, cfg: ClassicalPriorConfig) -> np.ndarray:
    """Binarize blackhat response with adaptive thresholding.
    Gates on minimum energy to avoid false positives on uniform images."""
    # If blackhat response is near-zero, no hair to detect
    if response.max() < 15:
        return np.zeros_like(response)
    # Global Otsu threshold first, then adaptive for fine detail
    _, global_mask = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        response, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        cfg.adaptive_block_size,
        cfg.adaptive_c,
    )
    # Intersection: only keep pixels detected by both methods
    return cv2.bitwise_and(global_mask, adaptive)


def _morphological_close(mask: np.ndarray, cfg: ClassicalPriorConfig) -> np.ndarray:
    """Close gaps in detected hair fragments."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.closing_kernel, cfg.closing_kernel)
    )
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=cfg.closing_iterations
    )


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin the mask to single-pixel-wide hair lines."""
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(mask)
    temp = mask.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skeleton


def extract_hair_mask(
    image: np.ndarray,
    config: ClassicalPriorConfig | None = None,
    return_skeleton: bool = False,
) -> np.ndarray:
    """
    Full Stage 1 pipeline: multi-scale blackhat → threshold → close → clean.
    Input: BGR image (H,W,3) uint8
    Output: binary mask (H,W) uint8, 255=hair
    """
    if config is None:
        config = ClassicalPriorConfig()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(gray)

    for ksize in config.kernel_sizes:
        response = _blackhat_at_scale(gray, ksize)
        binary = _adaptive_threshold(response, config)
        combined = cv2.bitwise_or(combined, binary)

    closed = _morphological_close(combined, config)
    cleaned = _remove_small_objects(closed, config.min_object_area)

    if return_skeleton:
        return _skeletonize(cleaned)
    return cleaned
