"""
Training loop for hair-robust classifier (Stage 5).
7-channel input: RGB + HairMask + DirectionalMap + FreqChannels.
Transfer learning with EfficientNet backbone.
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from app.ml.pipeline_config import PipelineConfig
from app.ml.models.robust_classifier import RobustClassifier


class ClassifierDataset(Dataset):
    """
    Dataset yielding 7-channel composed tensors + disease labels.
    Expects pre-composed .npy files (H,W,7) and integer labels.
    """

    def __init__(self, data_dir: str, label_file: str, size: int = 384):
        import numpy as np
        self.data_dir = Path(data_dir)
        self.size = size
        # label_file: each line is "filename label_int"
        self.samples = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np, cv2
        fname, label = self.samples[idx]
        data = np.load(self.data_dir / fname)  # (H,W,7) float32
        if data.shape[0] != self.size:
            data = cv2.resize(data, (self.size, self.size))
        tensor = torch.from_numpy(data.transpose(2, 0, 1)).float()
        return tensor, label


def train_classifier(data_dir: str, label_file: str):
    """Full classifier training."""
    cfg = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustClassifier(cfg.classifier).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs)
    ds = ClassifierDataset(data_dir, label_file, cfg.resolution.classifier_size)
    n_val = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, cfg.training.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, cfg.training.batch_size)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Acc: {correct/total:.3f} | Val: {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best_classifier.pth")

    print(f"Training complete. Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Hair-Robust Classifier")
    p.add_argument("--data", required=True, help="Dir with .npy composed tensors")
    p.add_argument("--labels", required=True, help="Label file (filename label)")
    args = p.parse_args()
    train_classifier(args.data, args.labels)
