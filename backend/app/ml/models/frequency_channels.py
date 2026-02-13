"""
Stage 4: Frequency Domain Channels.
Extracts high-frequency thin-structure features
that complement the spatial-domain detectors.
Pure OpenCV/NumPy — no deep learning.
"""
import cv2
import numpy as np
from app.ml.pipeline_config import FrequencyConfig


def _high_pass_filter(gray: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """Subtract Gaussian-blurred image from original to isolate high freq."""
    blurred = cv2.GaussianBlur(
        gray, (cfg.gaussian_ksize, cfg.gaussian_ksize), cfg.gaussian_sigma
    )
    high_pass = cv2.subtract(gray, blurred)
    return high_pass


def _laplacian_edge(gray: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """Laplacian edge detection for thin structures."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=cfg.laplacian_ksize)
    lap = np.abs(lap)
    # Normalize to 0-255
    lap_max = lap.max()
    if lap_max > 0:
        lap = (lap / lap_max * 255.0)
    return lap.astype(np.uint8)


def _frangi_vesselness(gray: np.ndarray, cfg: FrequencyConfig) -> np.ndarray:
    """
    Frangi vesselness filter — highlights tubular structures.
    Uses a multi-scale Hessian approach.
    """
    try:
        from skimage.filters import frangi
        gray_f = gray.astype(np.float64) / 255.0
        vessel = frangi(
            gray_f,
            sigmas=cfg.frangi_scales,
            black_ridges=True,
        )
        vessel = (vessel / (vessel.max() + 1e-7) * 255.0).astype(np.uint8)
        return vessel
    except ImportError:
        # Fallback: second derivative approximation
        return _laplacian_edge(gray, cfg)


def extract_frequency_channels(
    image: np.ndarray,
    config: FrequencyConfig | None = None,
) -> np.ndarray:
    """
    Full Stage 4: extract multiple frequency-domain feature channels.

    Input:  BGR image (H, W, 3) uint8
    Output: stacked channels (H, W, 3) uint8
            ch0 = high-pass, ch1 = laplacian edge, ch2 = frangi/laplacian
    """
    if config is None:
        config = FrequencyConfig()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ch0 = _high_pass_filter(gray, config)
    ch1 = _laplacian_edge(gray, config)

    if config.use_frangi:
        ch2 = _frangi_vesselness(gray, config)
    else:
        # Default: use Sobel magnitude as the third channel
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag_max = mag.max()
        if mag_max > 0:
            mag = (mag / mag_max * 255.0)
        ch2 = mag.astype(np.uint8)

    return np.stack([ch0, ch1, ch2], axis=-1)
