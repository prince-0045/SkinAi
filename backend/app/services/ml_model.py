"""
Skin Disease Prediction Service using DenseNet121.
"""

import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

# Constants
# ---------------------------------------------------------------------------
# Determine the project root (four levels up from 'ml_model.py')
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_H5_PATH = BASE_DIR / "model.h5"


_PKL_PATH = None  # No metadata pkl for this model
IMAGE_SIZE = 224
NUM_CLASSES = 7

# HAM10000 Class Labels
CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion"
]

# Medical information for the 7 HAM10000 classes
DISEASE_INFO = {
    "Actinic Keratosis": {
        "severity": "Moderate",
        "description": "Precancerous skin growth caused by sun damage. Often rough and scaly.",
        "recommendation": "Protect from sun and see a dermatologist for evaluation/treatment."
    },
    "Basal Cell Carcinoma": {
        "severity": "High",
        "description": "Slow-growing skin cancer. Rarely spreads but can damage surrounding tissue.",
        "recommendation": "Consult a dermatologist for surgical removal."
    },
    "Benign Keratosis": {
        "severity": "Low",
        "description": "Non-cancerous growths like seborrheic keratoses or lichen-planus like keratoses.",
        "recommendation": "Usually harmless. Monitor for changes."
    },
    "Dermatofibroma": {
        "severity": "Low",
        "description": "Common firm, noncancerous skin bumps.",
        "recommendation": "No treatment usually needed."
    },
    "Melanoma": {
        "severity": "Critical",
        "description": "Dangerous skin cancer in pigment-producing cells. High risk of spreading.",
        "recommendation": "IMMEDIATE medical attention required."
    },
    "Melanocytic Nevi": {
        "severity": "Low",
        "description": "Common mole. Typically benign pigment clusters.",
        "recommendation": "Monitor using ABCDE rule. Consult if changes occur."
    },
    "Vascular Lesion": {
        "severity": "Moderate",
        "description": "Abnormalities of blood vessels like angiomas.",
        "recommendation": "Consult a doctor if it bleeds or grows rapidly."
    },
    "Unknown": {
        "severity": "Unknown",
        "description": "The AI could not confidently identify the condition.",
        "recommendation": "Please provide a clearer photo or consult a medical professional."
    }
}

_model = None

def _get_model():
    """Load the MobileNet model lazily so the server starts fast."""
    global _model
    if _model is None:
        print(f"[ML] Rebuilding 7-class model and loading weights from {_H5_PATH} ...")
        
        # 1. Patch for Windows DLL Application Control policy block
        import sys
        import types
        try:
            mock_mod = types.ModuleType('tensorflow.python._pywrap_quantize_training')
            mock_mod.DoQuantizeTrainingOnGraphDefHelper = lambda *args, **kwargs: None
            sys.modules['tensorflow.python._pywrap_quantize_training'] = mock_mod
        except Exception:
            pass

        # 2. Import TF and legacy Keras
        import tensorflow as tf
        import tf_keras as keras
        
        # 3. Rebuild MobileNet architecture
        print("[ML] Constructing MobileNet backbone...")
        base_model = keras.applications.MobileNet(
            include_top=False,
            weights=None,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling='avg'
        )
        
        x = base_model.output
        # Note: Added Dropout and Dense to match trained head
        x = keras.layers.Dropout(0.2, name='dropout_1')(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='dense_1')(x)
        
        _model = keras.Model(inputs=base_model.input, outputs=outputs)
        
        # 4. Load weights by name
        print("[ML] Loading weights by name...")
        _model.load_weights(str(_H5_PATH), by_name=True, skip_mismatch=True)
        
        print("[ML] 7-Class MobileNet Model rebuilt and weights loaded successfully!")
    return _model


def predict(image_bytes: bytes):
    """Run real inference on skin image."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import tf_keras as keras

    # 1. Get the model
    model = _get_model()

    # 2. Preprocess Image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # MobileNet specific preprocessing (input usually in [-1, 1])
    img_array = keras.applications.mobilenet.preprocess_input(img_array)

    # 3. Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    disease = CLASS_NAMES[pred_idx]
    confidence = float(predictions[0][pred_idx])

    # 4. Get detailed info
    info = DISEASE_INFO.get(disease, DISEASE_INFO["Unknown"])

    return {
        "disease": disease,
        "confidence": confidence,
        "severity": info["severity"],
        "description": info["description"],
        "recommendation": info["recommendation"]
    }
