"""
Evaluation metrics for hair detection and classifier performance.
Covers: segmentation IoU/Dice/recall, border distortion, classification accuracy.
"""
import numpy as np
import torch


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over Union for binary masks."""
    pred_b = (pred > 127).astype(bool)
    target_b = (target > 127).astype(bool)
    intersection = (pred_b & target_b).sum()
    union = (pred_b | target_b).sum()
    return float(intersection / max(union, 1))


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient for binary masks."""
    pred_b = (pred > 127).astype(bool)
    target_b = (target > 127).astype(bool)
    intersection = (pred_b & target_b).sum()
    return float(2 * intersection / max(pred_b.sum() + target_b.sum(), 1))


def recall_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Hair detection recall: what fraction of true hair pixels are detected?"""
    pred_b = (pred > 127).astype(bool)
    target_b = (target > 127).astype(bool)
    tp = (pred_b & target_b).sum()
    return float(tp / max(target_b.sum(), 1))


def precision_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Hair detection precision."""
    pred_b = (pred > 127).astype(bool)
    target_b = (target > 127).astype(bool)
    tp = (pred_b & target_b).sum()
    return float(tp / max(pred_b.sum(), 1))


def border_distortion(
    hair_mask: np.ndarray, lesion_mask: np.ndarray
) -> float:
    """
    Measures overlap between hair mask and lesion border.
    Returns fraction: lower is better (target <5%).
    """
    import cv2
    # Lesion border = dilated - eroded
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(lesion_mask, kernel)
    eroded = cv2.erode(lesion_mask, kernel)
    border = cv2.subtract(dilated, eroded)

    hair_b = (hair_mask > 127).astype(bool)
    border_b = (border > 127).astype(bool)
    overlap = (hair_b & border_b).sum()
    return float(overlap / max(border_b.sum(), 1))


def classification_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate classifier accuracy.
    Returns: {accuracy, per_class_accuracy, total, correct}
    """
    model.eval()
    correct, total = 0, 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            for pred_i, target_i in zip(preds.cpu().tolist(), y.cpu().tolist()):
                class_total[target_i] = class_total.get(target_i, 0) + 1
                if pred_i == target_i:
                    class_correct[target_i] = class_correct.get(target_i, 0) + 1

    per_class = {k: class_correct.get(k, 0) / v for k, v in class_total.items()}
    return {"accuracy": correct / max(total, 1), "per_class": per_class, "total": total, "correct": correct}
