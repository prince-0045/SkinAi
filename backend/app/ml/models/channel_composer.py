"""
Stage 5a: Channel Composer.
Assembles the multi-channel input tensor for the robust classifier.
Merges: RGB(3) + HairMask(1) + DirectionalMap(1) + FreqChannels(2) = 7ch
Handles normalization and dtype consistency between heterogeneous sources.
"""
from __future__ import annotations
import numpy as np


def _normalize_single_channel(ch: np.ndarray) -> np.ndarray:
    """Normalize a single uint8 channel to [0, 1] float32."""
    return ch.astype(np.float32) / 255.0


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Squeeze any extra dims to ensure (H, W) shape."""
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def compose_channels(
    rgb: np.ndarray,
    hair_mask: np.ndarray,
    directional_map: np.ndarray,
    freq_channels: np.ndarray,
) -> np.ndarray:
    """
    Compose the 7-channel input array.

    Args:
        rgb:              (H, W, 3) uint8 BGR image
        hair_mask:        (H, W) uint8 binary mask (0 or 255)
        directional_map:  (H, W) uint8 response map
        freq_channels:    (H, W, 3) uint8 frequency channels

    Returns:
        composed: (H, W, 7) float32 normalized [0, 1]
    """
    rgb_f = rgb.astype(np.float32) / 255.0  # (H,W,3)
    mask_f = _normalize_single_channel(_ensure_2d(hair_mask))  # (H,W)
    dir_f = _normalize_single_channel(_ensure_2d(directional_map))  # (H,W)

    # Take first 2 freq channels to keep total at 7
    freq_f = freq_channels[..., :2].astype(np.float32) / 255.0  # (H,W,2)

    composed = np.concatenate([
        rgb_f,                              # ch 0,1,2
        mask_f[..., np.newaxis],            # ch 3
        dir_f[..., np.newaxis],             # ch 4
        freq_f,                             # ch 5,6
    ], axis=-1)
    return composed


def compose_to_tensor(
    rgb: np.ndarray,
    hair_mask: np.ndarray,
    directional_map: np.ndarray,
    freq_channels: np.ndarray,
) -> torch.Tensor:
    """
    Compose and convert to PyTorch tensor (C, H, W).
    Ready for batching with torch DataLoader.
    """
    composed = compose_channels(rgb, hair_mask, directional_map, freq_channels)
    import torch
    # HWC → CHW
    tensor = torch.from_numpy(composed.transpose(2, 0, 1))
    return tensor
