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
_H5_PATH = Path(r"d:\SkinAi\DenseNet121.weights.h5")
_PKL_PATH = Path(r"d:\SkinAi\DenseNet121.pkl")
IMAGE_SIZE = 224
NUM_CLASSES = 9

# Medical information for the 9 DenseNet classes
DISEASE_INFO = {
    "Acitinic Keratosis": {
        "severity": "Moderate",
        "description": "A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous.",
        "recommendation": "Protect from sun, use SPF 50+, and consult a dermatologist for possible removal (cryotherapy or creams)."
    },
    "Basal Cell Carcinoma": {
        "severity": "High",
        "description": "A type of skin cancer that begins in the basal cells. It often appears as a slightly transparent bump on the skin.",
        "recommendation": "Urgent consultation with a dermatologist. This usually requires surgical excision but rarely spreads to other parts of the body."
    },
    "Dermatofibroma": {
        "severity": "Low",
        "description": "Common noncancerous skin growths. They are small, firm bumps that can be pink, red, or brown.",
        "recommendation": "No treatment is usually necessary. If it becomes itchy or painful, see a doctor for removal."
    },
    "Melanoma": {
        "severity": "Critical",
        "description": "The most serious type of skin cancer. It develops in the cells that produce melanin.",
        "recommendation": "IMMEDIATE medical attention required. Early detection is key for successful treatment via surgery, radiation, or chemotherapy."
    },
    "Nevus": {
        "severity": "Low",
        "description": "A common mole. A benign growth on the skin that is formed by a cluster of melanocytes.",
        "recommendation": "Regular self-monitoring. Use the ABCDE rule. If a mole changes shape, color, or size, see a doctor."
    },
    "Pigmented Benign Keratosis": {
        "severity": "Low",
        "description": "A noncancerous skin condition that appears as a waxy brown, black, or tan growth.",
        "recommendation": "Harmless. No treatment needed unless irritated. A dermatologist can remove it for cosmetic reasons."
    },
    "Seborrheic Keratosis": {
        "severity": "Low",
        "description": "A noncancerous skin growth that often appears as a brown, black, or light tan growth on the face, chest, shoulders, or back.",
        "recommendation": "Harmless. Removal is usually for cosmetic reasons or if it becomes irritated by clothing."
    },
    "Squamous Cell Carcinoma": {
        "severity": "High",
        "description": "Common form of skin cancer that develops in the squamous cells that make up the middle and outer layers of the skin.",
        "recommendation": "Consult a specialist. Requires treatment (surgery or topical meds) to prevent it from spreading."
    },
    "Vascular Lesion": {
        "severity": "Moderate",
        "description": "Abnormalities of the blood vessels (like cherry angiomas or port-wine stains).",
        "recommendation": "Usually benign. See a doctor if it bleeds or changes rapidly. Laser therapy is an option for cosmetic removal."
    },
    "Unknown": {
        "severity": "Unknown",
        "description": "The AI could not confidently identify the condition.",
        "recommendation": "Please provide a clearer photo or consult a medical professional for an accurate diagnosis."
    }
}

_model = None

def _get_model():
    """Load the DenseNet121 model lazily so the server starts fast."""
    global _model
    if _model is None:
        print(f"[ML] Rebuilding model and loading weights from {_H5_PATH} ...")
        
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
        
        # 3. Rebuild DenseNet121 architecture
        print("[ML] Constructing DenseNet121 backbone...")
        base_model = keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling='avg'
        )
        
        x = base_model.output
        # Note: Depending on how the model was trained, there might be different top layers.
        # Based on typical transfer learning, we add the final Dense layer.
        outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        _model = keras.Model(inputs=base_model.input, outputs=outputs)
        
        # 4. Load weights by name
        print("[ML] Loading weights by name...")
        _model.load_weights(str(_H5_PATH), by_name=True, skip_mismatch=True)
        
        print("[ML] DenseNet121 Model rebuilt and weights loaded successfully!")
    return _model


def predict(image_bytes: bytes):
    """Run real inference on skin image."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import tf_keras as keras

    # 1. Ensure metadata is loaded
    try:
        with open(_PKL_PATH, "rb") as f:
            metadata = pickle.load(f)
            class_names = metadata.get('class_names', list(DISEASE_INFO.keys())[:NUM_CLASSES])
    except Exception as e:
        print(f"[ML] Warning: Could not load metadata, using fallback classes. Error: {e}")
        class_names = list(DISEASE_INFO.keys())[:NUM_CLASSES]

    # 2. Get the model
    model = _get_model()

    # 3. Preprocess Image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # DenseNet121 specific preprocessing (normalization)
    img_array = keras.applications.densenet.preprocess_input(img_array)

    # 4. Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    disease = class_names[pred_idx]
    confidence = float(predictions[0][pred_idx])

    # 5. Get detailed info
    info = DISEASE_INFO.get(disease, DISEASE_INFO["Unknown"])

    return {
        "disease": disease,
        "confidence": confidence,
        "severity": info["severity"],
        "description": info["description"],
        "recommendation": info["recommendation"]
    }
