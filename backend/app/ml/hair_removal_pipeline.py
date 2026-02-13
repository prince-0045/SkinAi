"""
End-to-end hair removal inference pipeline.
Sequences all stages: classical prior → deep detector → directional →
frequency channels → channel composition.
Lazy-loads deep models. Falls back to classical-only mode.
"""
import cv2
import numpy as np
from pathlib import Path
from app.ml.pipeline_config import PipelineConfig
from app.ml.models.classical_prior import extract_hair_mask
from app.ml.models.directional_filters import extract_directional_map
from app.ml.models.frequency_channels import extract_frequency_channels
from app.ml.models.channel_composer import compose_channels


class HairRemovalPipeline:
    """Orchestrates multi-stage hair detection and classifier integration."""

    def __init__(self, config: PipelineConfig | None = None, model_path: str | None = None):
        self.cfg = config or PipelineConfig()
        self._deep_model = None
        self._model_path = model_path

    def _load_deep_model(self):
        """Lazy-load the trained deep detector if weights exist."""
        if self._deep_model is not None:
            return self._deep_model
        try:
            import torch
            from app.ml.models.hair_detector import HairDetector
            path = self._model_path or Path(self.cfg.training.checkpoint_dir) / "best_hair_detector.pth"
            if Path(path).exists():
                model = HairDetector(self.cfg.detector)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model.eval()
                self._deep_model = model
                return model
        except Exception:
            pass
        return None

    def _enforce_resolution(self, image: np.ndarray) -> np.ndarray:
        """Ensure minimum resolution for detection stage."""
        h, w = image.shape[:2]
        min_size = self.cfg.resolution.min_detection_size
        if min(h, w) < min_size:
            scale = min_size / min(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        return image

    def _deep_detect(self, image: np.ndarray) -> np.ndarray | None:
        """Run deep detector if available. Returns binary mask or None."""
        model = self._load_deep_model()
        if model is None:
            return None
        import torch
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        return (model.predict_mask(tensor).squeeze().numpy() * 255).astype(np.uint8)

    def process(self, image: np.ndarray) -> dict:
        """
        Run full pipeline on a BGR image.
        Returns dict with: composed, hair_mask, directional, frequency, original
        """
        image = self._enforce_resolution(image)

        # Stage 1: Classical prior
        classical_mask = extract_hair_mask(image, self.cfg.classical)

        # Stage 2: Deep detector (optional)
        deep_mask = self._deep_detect(image)
        if deep_mask is not None:
            # Union of classical + deep for best recall
            hair_mask = cv2.bitwise_or(classical_mask, deep_mask)
        else:
            hair_mask = classical_mask

        # Stage 3: Directional enhancement
        directional = extract_directional_map(image, self.cfg.directional)

        # Stage 4: Frequency channels
        freq = extract_frequency_channels(image, self.cfg.frequency)

        # Stage 5: Compose
        composed = compose_channels(image, hair_mask, directional, freq)

        return {
            "composed": composed,
            "hair_mask": hair_mask,
            "directional": directional,
            "frequency": freq,
            "original": image,
        }
