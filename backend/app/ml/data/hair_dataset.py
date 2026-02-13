"""
Data pipeline: Hair Segmentation Dataset.
Supports both paired (image, mask) data and pseudo-label mode
where classical prior generates masks on-the-fly.
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from app.ml.pipeline_config import ResolutionConfig


class HairSegmentationDataset(Dataset):
    """
    PyTorch dataset for hair segmentation.

    Modes:
      - paired: image_dir + mask_dir with matching filenames
      - pseudo: image_dir only, masks generated via classical prior
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        resolution: ResolutionConfig | None = None,
        transform=None,
    ):
        self.resolution = resolution or ResolutionConfig()
        self.transform = transform

        self.image_paths = sorted(Path(image_dir).glob("*"))
        self.image_paths = [p for p in self.image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        if mask_dir:
            self.mask_paths = sorted(Path(mask_dir).glob("*"))
            self.mask_paths = [p for p in self.mask_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        else:
            self.mask_paths = None

        self._classical_prior = None

    def _get_classical_prior(self):
        """Lazy import to avoid circular deps."""
        if self._classical_prior is None:
            from app.ml.models.classical_prior import extract_hair_mask
            self._classical_prior = extract_hair_mask
        return self._classical_prior

    def _load_and_resize(self, path: Path, is_mask: bool = False) -> np.ndarray:
        """Load image and enforce minimum resolution."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        size = self.resolution.min_detection_size
        h, w = img.shape[:2]
        if min(h, w) < size:
            scale = size / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        # Center crop to square
        h, w = img.shape[:2]
        s = min(h, w)
        y, x = (h - s) // 2, (w - s) // 2
        img = img[y:y+s, x:x+s]
        return cv2.resize(img, (size, size))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = self._load_and_resize(self.image_paths[idx])

        if self.mask_paths:
            mask = self._load_and_resize(self.mask_paths[idx], is_mask=True)
        else:
            extract_fn = self._get_classical_prior()
            mask = extract_fn(image)

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image, mask = result["image"], result["mask"]

        img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return {"image": img_t, "mask": mask_t, "path": str(self.image_paths[idx])}
