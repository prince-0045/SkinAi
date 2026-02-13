"""
Test full inference pipeline in classical-only mode (no trained models).
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_pipeline_classical_only():
    """Pipeline should work without any trained deep models."""
    from app.ml.hair_removal_pipeline import HairRemovalPipeline

    pipeline = HairRemovalPipeline()

    # Create a synthetic dermoscopic-like image
    img = np.random.randint(140, 200, (300, 300, 3), dtype=np.uint8)
    # Add synthetic hair lines
    for i in range(5):
        y = 40 + i * 50
        cv2.line(img, (10, y), (290, y + 15), (30, 30, 30), 2)

    result = pipeline.process(img)

    # Verify all expected keys
    expected_keys = {"composed", "hair_mask", "directional", "frequency", "original"}
    assert set(result.keys()) == expected_keys, f"Keys: {result.keys()}"

    # Verify shapes (pipeline enforces min 512x512)
    h, w = result["original"].shape[:2]
    assert min(h, w) >= 512, f"Resolution too low: {h}x{w}"
    assert result["composed"].shape == (h, w, 7), f"Composed shape: {result['composed'].shape}"
    assert result["hair_mask"].shape == (h, w), f"Mask shape: {result['hair_mask'].shape}"
    assert result["directional"].shape == (h, w), f"Dir shape: {result['directional'].shape}"
    assert result["frequency"].shape == (h, w, 3), f"Freq shape: {result['frequency'].shape}"
    print(f"  PASS: Pipeline output shapes correct ({h}x{w})")


def test_pipeline_composed_values():
    """Composed tensor values should be normalized [0, 1]."""
    from app.ml.hair_removal_pipeline import HairRemovalPipeline

    pipeline = HairRemovalPipeline()
    img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    result = pipeline.process(img)

    composed = result["composed"]
    assert composed.dtype == np.float32, f"Wrong dtype: {composed.dtype}"
    assert composed.min() >= 0, f"Min below 0: {composed.min()}"
    assert composed.max() <= 1.0, f"Max above 1: {composed.max()}"
    print("  PASS: Composed values normalized correctly")


def test_directional_filter_standalone():
    """Directional filter should produce a 2D response map."""
    from app.ml.models.directional_filters import extract_directional_map

    img = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
    cv2.line(img, (0, 64), (127, 64), (30, 30, 30), 2)  # horizontal hair
    dmap = extract_directional_map(img)

    assert dmap.shape == (128, 128), f"Shape: {dmap.shape}"
    assert dmap.dtype == np.uint8
    print(f"  PASS: Directional map range [{dmap.min()}, {dmap.max()}]")


def test_frequency_channels_standalone():
    """Frequency channels should produce (H,W,3) output."""
    from app.ml.models.frequency_channels import extract_frequency_channels

    img = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
    freq = extract_frequency_channels(img)

    assert freq.shape == (128, 128, 3), f"Shape: {freq.shape}"
    assert freq.dtype == np.uint8
    print("  PASS: Frequency channels output correct")


if __name__ == "__main__":
    print("=== Test: Full Pipeline ===")
    test_pipeline_classical_only()
    test_pipeline_composed_values()
    test_directional_filter_standalone()
    test_frequency_channels_standalone()
    print("All pipeline tests PASSED.\n")
