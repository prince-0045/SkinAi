"""
Pytest tests for SkinAi backend.
Tests ML prediction logic and API auth endpoints.
"""
import pytest
from app.services.ml_model import predict, CLASS_NAMES, DISEASE_INFO, MERGED_CLASS_INCLUDES


class TestMLPrediction:
    """Tests for the ML prediction pipeline."""

    def test_predict_returns_valid_category(self, sample_image_bytes):
        """Model must return a category that exists in DISEASE_INFO."""
        result = predict(sample_image_bytes)
        assert result["category"] in DISEASE_INFO, f"Unknown category: {result['category']}"

    def test_predict_returns_required_fields(self, sample_image_bytes):
        """API contract: all required fields must be present."""
        result = predict(sample_image_bytes)
        required_fields = [
            "category", "disease", "confidence", "is_unknown",
            "includes", "severity", "description", "recommendation",
            "do_list", "dont_list"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_predict_confidence_range(self, sample_image_bytes):
        """Confidence must be between 0 and 1."""
        result = predict(sample_image_bytes)
        assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of range: {result['confidence']}"

    def test_predict_is_unknown_flag(self, sample_image_bytes):
        """is_unknown must be a boolean."""
        result = predict(sample_image_bytes)
        assert isinstance(result["is_unknown"], bool)

    def test_predict_severity_values(self, sample_image_bytes):
        """Severity must be one of the expected levels."""
        result = predict(sample_image_bytes)
        valid_severities = {"Mild", "Moderate", "Severe", "Low", "N/A"}
        assert result["severity"] in valid_severities, f"Invalid severity: {result['severity']}"

    def test_predict_lists_are_lists(self, sample_image_bytes):
        """do_list and dont_list must be lists."""
        result = predict(sample_image_bytes)
        assert isinstance(result["do_list"], list)
        assert isinstance(result["dont_list"], list)

    def test_predict_random_noise_low_confidence(self):
        """Random noise image should not return high confidence for any real disease."""
        import numpy as np
        from PIL import Image
        from io import BytesIO

        noise = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(noise).save(buf, format="JPEG")
        result = predict(buf.getvalue())
        # Either low confidence or Unknown
        assert result["confidence"] < 0.8 or result["category"] == "Unknown"

    def test_disease_info_completeness(self):
        """Every disease category must have complete info."""
        for category, info in DISEASE_INFO.items():
            assert "severity" in info, f"{category} missing severity"
            assert "description" in info, f"{category} missing description"
            assert "recommendation" in info, f"{category} missing recommendation"
            assert "do_list" in info, f"{category} missing do_list"
            assert "dont_list" in info, f"{category} missing dont_list"

    def test_class_names_count(self):
        """Verify expected number of class names."""
        assert len(CLASS_NAMES) > 0, "CLASS_NAMES is empty"

    def test_merged_classes_valid(self):
        """All merged class targets must exist in DISEASE_INFO."""
        for merged_name, includes in MERGED_CLASS_INCLUDES.items():
            assert merged_name in DISEASE_INFO, f"Merged class {merged_name} not in DISEASE_INFO"
