"""
Test loss functions: verify combined loss produces finite, non-zero scalar.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_combined_loss_basic():
    """CombinedSegmentationLoss should return a finite positive scalar."""
    from app.ml.training.loss_functions import CombinedSegmentationLoss

    criterion = CombinedSegmentationLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    result = criterion(pred, target)

    assert "total" in result, "Missing 'total' key"
    assert isinstance(result["total"], torch.Tensor), "total not a tensor"
    assert torch.isfinite(result["total"]), f"total not finite: {result['total']}"
    assert result["total"].item() > 0, f"total should be positive: {result['total']}"
    print(f"  PASS: Combined loss = {result['total'].item():.4f}")


def test_individual_losses():
    """Each component loss should be a finite float."""
    from app.ml.training.loss_functions import CombinedSegmentationLoss

    criterion = CombinedSegmentationLoss()
    pred = torch.randn(4, 1, 32, 32)
    target = torch.randint(0, 2, (4, 1, 32, 32)).float()

    result = criterion(pred, target)

    for key in ["bce", "dice", "focal", "edge"]:
        assert key in result, f"Missing key: {key}"
        val = result[key]
        assert isinstance(val, float), f"{key} not float"
        assert not (val != val), f"{key} is NaN"  # NaN check
        print(f"  PASS: {key} = {val:.4f}")


def test_loss_gradient_flow():
    """Verify gradients flow through the combined loss."""
    from app.ml.training.loss_functions import CombinedSegmentationLoss

    criterion = CombinedSegmentationLoss()
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    result = criterion(pred, target)
    result["total"].backward()

    assert pred.grad is not None, "No gradients!"
    assert torch.isfinite(pred.grad).all(), "Gradient contains inf/nan"
    print("  PASS: Gradients flow correctly")


if __name__ == "__main__":
    print("=== Test: Loss Functions ===")
    test_combined_loss_basic()
    test_individual_losses()
    test_loss_gradient_flow()
    print("All loss function tests PASSED.\n")
