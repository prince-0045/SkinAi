"""
Pytest test configuration and shared fixtures.
"""
import pytest
import sys
import os

# Add backend root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def sample_image_bytes():
    """Generate a simple test image (solid color skin-like patch)."""
    from PIL import Image
    from io import BytesIO
    import numpy as np

    # Create a skin-toned image (more realistic than random noise)
    skin_color = np.full((224, 224, 3), [180, 140, 120], dtype=np.uint8)
    # Add some variation
    noise = np.random.randint(-20, 20, skin_color.shape, dtype=np.int16)
    img_array = np.clip(skin_color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
