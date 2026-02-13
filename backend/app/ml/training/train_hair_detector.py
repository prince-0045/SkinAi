"""
Training loop for deep hair detector (Stage 2).
Colab-ready. Config-driven. Supports pseudo-label training.
"""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from app.ml.pipeline_config import PipelineConfig
from app.ml.models.hair_detector import HairDetector
from app.ml.training.loss_functions import CombinedSegmentationLoss
from app.ml.data.hair_dataset import HairSegmentationDataset


def create_dataloaders(cfg: PipelineConfig, image_dir: str, mask_dir: str | None):
    """Build train/val dataloaders."""
    ds = HairSegmentationDataset(image_dir, mask_dir, cfg.resolution)
    n_val = max(1, int(len(ds) * 0.15))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_dl = DataLoader(train_ds, cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_dl = DataLoader(val_ds, cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers)
    return train_dl, val_dl


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Single training epoch. Returns avg loss dict."""
    model.train()
    totals = {}
    for batch in loader:
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits, _ = model(imgs)
            losses = criterion(logits, masks)
        loss = losses["total"]
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        for k, v in losses.items():
            totals[k] = totals.get(k, 0) + (v if isinstance(v, float) else v.item())
    return {k: v / len(loader) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation pass. Returns avg loss dict."""
    model.eval()
    totals = {}
    for batch in loader:
        logits, _ = model(batch["image"].to(device))
        losses = criterion(logits, batch["mask"].to(device))
        for k, v in losses.items():
            totals[k] = totals.get(k, 0) + (v if isinstance(v, float) else v.item())
    return {k: v / len(loader) for k, v in totals.items()}


def train(image_dir: str, mask_dir: str | None = None):
    """Full training pipeline."""
    cfg = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HairDetector(cfg.detector).to(device)
    criterion = CombinedSegmentationLoss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs)
    scaler = torch.amp.GradScaler("cuda") if cfg.training.mixed_precision and device.type == "cuda" else None
    train_dl, val_dl = create_dataloaders(cfg, image_dir, mask_dir)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg.training.epochs):
        t_loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, device)
        v_loss = validate(model, val_dl, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{cfg.training.epochs} | Train: {t_loss['total']:.4f} | Val: {v_loss['total']:.4f}")
        if v_loss["total"] < best_loss:
            best_loss = v_loss["total"]
            torch.save(model.state_dict(), ckpt_dir / "best_hair_detector.pth")

    torch.save(model.state_dict(), ckpt_dir / "final_hair_detector.pth")
    print(f"Training complete. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hair Detector")
    parser.add_argument("--images", required=True, help="Path to image directory")
    parser.add_argument("--masks", default=None, help="Path to mask directory (optional)")
    args = parser.parse_args()
    train(args.images, args.masks)
