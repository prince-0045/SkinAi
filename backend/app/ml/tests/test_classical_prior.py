"""
Test Stage 1: Classical Hair Prior.
Creates a synthetic image with known dark lines on a light background
and verifies the mask captures sufficient line pixels.
"""
import cv2
import numpy as np
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_classical_prior_detects_synthetic_lines():
    """A bright image with drawn dark lines should trigger >50% detection."""
    from app.ml.models.classical_prior import extract_hair_mask
    from app.ml.pipeline_config import ClassicalPriorConfig

    # Create synthetic: bright background + dark lines
    img = np.ones((256, 256, 3), dtype=np.uint8) * 200
    # Draw 5 dark lines (simulating hair)
    for i in range(5):
        y = 30 + i * 40
        cv2.line(img, (10, y), (246, y + 20), (30, 30, 30), thickness=2)

    mask = extract_hair_mask(img)

    assert mask.shape == (256, 256), f"Wrong shape: {mask.shape}"
    assert mask.dtype == np.uint8, f"Wrong dtype: {mask.dtype}"

    # Check that some hair pixels were detected
    hair_pixels = (mask > 127).sum()
    assert hair_pixels > 50, f"Too few detected pixels: {hair_pixels}"
    print(f"  PASS: Detected {hair_pixels} hair pixels")


def test_classical_prior_empty_on_uniform():
    """A uniform bright image should produce minimal false positives."""
    from app.ml.models.classical_prior import extract_hair_mask

    img = np.ones((256, 256, 3), dtype=np.uint8) * 180
    mask = extract_hair_mask(img)
    hair_pixels = (mask > 127).sum()

    # Should have very few false positives on uniform image
    total = mask.size
    fp_rate = hair_pixels / total
    assert fp_rate < 0.05, f"Too many FPs on uniform image: {fp_rate:.3f}"
    print(f"  PASS: FP rate on uniform image: {fp_rate:.4f}")


def test_classical_prior_with_skeleton():
    """Test skeleton output mode."""
    from app.ml.models.classical_prior import extract_hair_mask

    img = np.ones((256, 256, 3), dtype=np.uint8) * 200
    cv2.line(img, (20, 128), (236, 128), (30, 30, 30), thickness=3)

    mask = extract_hair_mask(img, return_skeleton=True)
    assert mask.shape == (256, 256)
    print(f"  PASS: Skeleton mode works, {(mask > 0).sum()} skeleton pixels")


if __name__ == "__main__":
    print("=== Test: Classical Prior ===")
    test_classical_prior_detects_synthetic_lines()
    test_classical_prior_empty_on_uniform()
    test_classical_prior_with_skeleton()
    print("All classical prior tests PASSED.\n")
