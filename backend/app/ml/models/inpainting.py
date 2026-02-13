"""
Optional Stage 6: Controlled Inpainting.
Partial convolution + edge-aware interpolation for cases
where explicit hair removal is required. No GAN.
"""
import cv2
import numpy as np


def _dilate_mask(mask: np.ndarray, dilation_px: int = 3) -> np.ndarray:
    """Dilate hair mask slightly to cover hair edges."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


def inpaint_telea(
    image: np.ndarray, mask: np.ndarray, radius: int = 5
) -> np.ndarray:
    """
    Fast marching inpainting (Telea method).
    Input:  BGR image (H,W,3), binary mask (H,W) [255=inpaint region]
    Output: inpainted BGR image (H,W,3)
    """
    dilated = _dilate_mask(mask)
    return cv2.inpaint(image, dilated, radius, cv2.INPAINT_TELEA)


def inpaint_ns(
    image: np.ndarray, mask: np.ndarray, radius: int = 5
) -> np.ndarray:
    """
    Navier-Stokes inpainting — better for thin structures.
    Input:  BGR image (H,W,3), binary mask (H,W) [255=inpaint region]
    Output: inpainted BGR image (H,W,3)
    """
    dilated = _dilate_mask(mask)
    return cv2.inpaint(image, dilated, radius, cv2.INPAINT_NS)


def edge_aware_interpolation(
    image: np.ndarray, mask: np.ndarray, d: int = 9
) -> np.ndarray:
    """
    For each masked pixel, use bilateral-filtered values from neighbors.
    Preserves edges better than Gaussian interpolation.
    """
    bilateral = cv2.bilateralFilter(image, d, 75, 75)
    result = image.copy()
    mask_bool = mask > 127
    result[mask_bool] = bilateral[mask_bool]
    return result


def controlled_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "ns",
    radius: int = 5,
) -> np.ndarray:
    """
    Dispatcher for inpainting strategies.
    Methods: 'telea', 'ns', 'bilateral'
    """
    if method == "telea":
        return inpaint_telea(image, mask, radius)
    elif method == "ns":
        return inpaint_ns(image, mask, radius)
    elif method == "bilateral":
        return edge_aware_interpolation(image, mask)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")


def inpaint_with_confidence(
    image: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.7,
    method: str = "ns",
) -> np.ndarray:
    """
    Only inpaint pixels where detection confidence exceeds threshold.
    Prevents over-aggressive removal of uncertain regions.
    """
    confident_mask = np.zeros_like(mask)
    confident_mask[(mask > 127) & (confidence > threshold * 255)] = 255
    return controlled_inpaint(image, confident_mask, method)
