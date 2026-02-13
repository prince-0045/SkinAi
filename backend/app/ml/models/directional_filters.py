"""
Stage 3: Directional Enhancement via Gabor Filter Bank.
Captures curved and rotated hair strands that axis-aligned
convolutions miss. Pure NumPy/OpenCV — no deep learning.
"""
import cv2
import numpy as np
from app.ml.pipeline_config import DirectionalConfig


def _build_gabor_bank(cfg: DirectionalConfig) -> list[np.ndarray]:
    """Build a bank of Gabor kernels at multiple angles and frequencies."""
    kernels = []
    angles = np.linspace(0, np.pi, cfg.num_angles, endpoint=False)
    for theta in angles:
        for freq in cfg.frequencies:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=cfg.sigma,
                theta=theta,
                lambd=1.0 / freq,
                gamma=cfg.gamma,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernel /= kernel.sum() + 1e-7
            kernels.append(kernel)
    return kernels


def _apply_gabor_bank(
    gray: np.ndarray, kernels: list[np.ndarray]
) -> np.ndarray:
    """Apply all Gabor kernels and take per-pixel maximum response."""
    responses = np.stack(
        [cv2.filter2D(gray, cv2.CV_32F, k) for k in kernels], axis=0
    )
    return np.max(responses, axis=0)


def _normalize_response(response: np.ndarray) -> np.ndarray:
    """Normalize response to [0, 255] uint8."""
    rmin, rmax = response.min(), response.max()
    if rmax - rmin < 1e-6:
        return np.zeros_like(response, dtype=np.uint8)
    normalized = (response - rmin) / (rmax - rmin) * 255.0
    return normalized.astype(np.uint8)


def extract_directional_map(
    image: np.ndarray,
    config: DirectionalConfig | None = None,
) -> np.ndarray:
    """
    Full Stage 3 pipeline.
    Input:  BGR image (H, W, 3) uint8
    Output: directional response map (H, W) uint8, bright = strong hair signal
    """
    if config is None:
        config = DirectionalConfig()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernels = _build_gabor_bank(config)
    response = _apply_gabor_bank(gray, kernels)
    return _normalize_response(response)


def extract_dominant_angle(
    image: np.ndarray,
    config: DirectionalConfig | None = None,
) -> np.ndarray:
    """
    Returns per-pixel dominant orientation angle (in radians).
    Useful for visualization or rotation-aware post-processing.
    """
    if config is None:
        config = DirectionalConfig()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    angles = np.linspace(0, np.pi, config.num_angles, endpoint=False)
    all_responses = []
    for theta in angles:
        for freq in config.frequencies:
            kernel = cv2.getGaborKernel(
                (21, 21), config.sigma, theta, 1.0 / freq, config.gamma
            )
            r = cv2.filter2D(gray, cv2.CV_32F, kernel / (kernel.sum() + 1e-7))
            all_responses.append((theta, r))

    best_angle = np.zeros_like(gray)
    best_val = np.full_like(gray, -np.inf)
    for theta, r in all_responses:
        mask = r > best_val
        best_angle[mask] = theta
        best_val[mask] = r[mask]
    return best_angle
