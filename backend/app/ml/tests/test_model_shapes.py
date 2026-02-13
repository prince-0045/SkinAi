"""
Test model forward passes: verify output shapes for all key models.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_hair_detector_shapes():
    """HairDetector should produce correct output shapes."""
    from app.ml.models.hair_detector import HairDetector

    model = HairDetector()
    x = torch.randn(2, 3, 128, 128)
    logits, conf = model(x)

    assert logits.shape == (2, 1, 128, 128), f"Logits shape: {logits.shape}"
    assert conf.shape == (2, 1, 128, 128), f"Conf shape: {conf.shape}"
    assert conf.min() >= 0 and conf.max() <= 1, "Confidence not in [0,1]"
    print(f"  PASS: HairDetector shapes correct. Params: {HairDetector.count_parameters(model):,}")


def test_hair_detector_predict_mask():
    """predict_mask should return binary uint8 tensor."""
    from app.ml.models.hair_detector import HairDetector

    model = HairDetector()
    x = torch.randn(1, 3, 64, 64)
    binary = model.predict_mask(x)

    assert binary.dtype == torch.uint8, f"Wrong dtype: {binary.dtype}"
    assert set(binary.unique().tolist()).issubset({0, 1}), "Not binary"
    print("  PASS: predict_mask returns binary uint8")


def test_robust_classifier_shapes():
    """RobustClassifier should accept 7-channel input and output class logits."""
    from app.ml.models.robust_classifier import RobustClassifier
    from app.ml.pipeline_config import ClassifierConfig

    cfg = ClassifierConfig(pretrained=False)  # No download in tests
    model = RobustClassifier(cfg)
    x = torch.randn(2, 7, 224, 224)
    out = model(x)

    assert out.shape == (2, 5), f"Output shape: {out.shape}"
    print(f"  PASS: RobustClassifier output shape correct")


def test_channel_composer():
    """Channel composer should produce (H,W,7) array."""
    import numpy as np
    from app.ml.models.channel_composer import compose_channels

    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    directional = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    freq = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    composed = compose_channels(rgb, mask, directional, freq)
    assert composed.shape == (64, 64, 7), f"Wrong shape: {composed.shape}"
    assert composed.dtype == np.float32
    assert composed.min() >= 0 and composed.max() <= 1.0
    print("  PASS: compose_channels output correct")


if __name__ == "__main__":
    print("=== Test: Model Shapes ===")
    test_hair_detector_shapes()
    test_hair_detector_predict_mask()
    test_robust_classifier_shapes()
    test_channel_composer()
    print("All model shape tests PASSED.\n")
