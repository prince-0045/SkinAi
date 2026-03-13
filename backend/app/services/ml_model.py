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
# Determine the backend root (three levels up from 'ml_model.py')
BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
        "description": "A precancerous rough, scaly patch on the skin caused by years of sun exposure. It appears as a dry, flaky patch that may itch or burn. Early treatment is important to prevent progression to skin cancer.",
        "recommendation": "See a dermatologist for evaluation and possible cryotherapy, topical creams, or photodynamic therapy.",
        "do_list": [
            "Apply broad-spectrum SPF 30+ sunscreen daily",
            "Wear protective clothing and wide-brimmed hats outdoors",
            "Visit a dermatologist for professional evaluation and treatment",
            "Perform regular self-skin checks for new or changing patches",
            "Keep affected skin moisturized"
        ],
        "dont_list": [
            "Do not scratch or pick at the rough patches",
            "Avoid prolonged direct sun exposure especially between 10am–4pm",
            "Do not use tanning beds or UV lamps",
            "Do not ignore it — untreated it can progress to skin cancer",
            "Avoid harsh exfoliants on affected areas"
        ]
    },
    "Basal Cell Carcinoma": {
        "severity": "High",
        "description": "The most common form of skin cancer, arising from basal cells in the deepest layer of the epidermis. It grows slowly and rarely spreads, but can cause significant local tissue damage if left untreated. Often appears as a pearly or waxy bump.",
        "recommendation": "Consult a dermatologist promptly for surgical removal, Mohs surgery, or radiation therapy.",
        "do_list": [
            "See a dermatologist or oncologist as soon as possible",
            "Follow the prescribed treatment plan (surgery, radiation, topical agents)",
            "Apply sunscreen SPF 50+ every day",
            "Protect the affected area from sun exposure",
            "Attend all follow-up appointments to monitor for recurrence"
        ],
        "dont_list": [
            "Do not delay medical treatment — early removal is critical",
            "Do not expose the lesion to direct sunlight",
            "Do not attempt to remove or treat it yourself at home",
            "Avoid tanning beds and UV exposure entirely",
            "Do not miss follow-up appointments"
        ]
    },
    "Benign Keratosis": {
        "severity": "Low",
        "description": "A non-cancerous skin growth including seborrheic keratoses and lichen-planus-like keratoses. These typically appear as brown, black, or tan waxy growths. They are harmless but can be cosmetically bothersome.",
        "recommendation": "Usually harmless. Monitor for changes, and consult a doctor if you want removal for cosmetic reasons.",
        "do_list": [
            "Monitor the growth for any changes in size, color, or shape",
            "Consult a doctor if it bleeds, itches excessively, or grows rapidly",
            "Seek removal if it causes discomfort or is cosmetically concerning",
            "Keep skin moisturized to reduce irritation",
            "Get a professional diagnosis to confirm it's benign"
        ],
        "dont_list": [
            "Do not pick, scratch, or attempt to remove it yourself",
            "Do not use over-the-counter wart removers without medical advice",
            "Do not ignore sudden changes in appearance",
            "Avoid harsh skin products that irritate the area"
        ]
    },
    "Dermatofibroma": {
        "severity": "Low",
        "description": "A common, firm, non-cancerous skin nodule most often found on the legs. It results from a minor injury or insect bite and is composed of fibrous tissue. It may be slightly tender when pressed.",
        "recommendation": "No treatment is usually needed. Consult a doctor if it grows or becomes painful.",
        "do_list": [
            "Leave it alone if it causes no discomfort",
            "Consult a dermatologist if it grows rapidly or changes",
            "Protect the area from repeated trauma or pressure",
            "Use gentle skincare products around the area"
        ],
        "dont_list": [
            "Do not try to pop or squeeze it",
            "Do not cut or attempt home removal",
            "Avoid shaving directly over it to prevent irritation",
            "Do not ignore any sudden change in size or color"
        ]
    },
    "Melanoma": {
        "severity": "Critical",
        "description": "The most serious type of skin cancer, developing in melanocytes (pigment-producing cells). It can spread to other organs rapidly if not caught early. Looks like an unusual mole with irregular borders, multiple colors, or rapid growth.",
        "recommendation": "⚠️ IMMEDIATE medical attention required. Contact a dermatologist or oncologist TODAY.",
        "do_list": [
            "Seek immediate medical attention — do not wait",
            "Document and photograph the lesion to track any changes",
            "Get a biopsy performed by a professional as soon as possible",
            "Follow your doctor's treatment plan (surgery, immunotherapy, targeted therapy)",
            "Notify immediate family members as there may be hereditary risk",
            "Use extreme sun protection for all exposed skin"
        ],
        "dont_list": [
            "Do NOT delay seeking medical care — every day matters",
            "Do not expose the lesion to any sunlight or UV radiation",
            "Do not attempt any home remedies or self-treatment",
            "Do not ignore it or assume it will go away on its own",
            "Avoid tanning beds permanently",
            "Do not miss any oncology follow-up appointments"
        ]
    },
    "Melanocytic Nevi": {
        "severity": "Low",
        "description": "A common benign mole formed by a cluster of melanocytes (pigment-producing cells). Most are harmless but should be regularly monitored for changes using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution).",
        "recommendation": "Monitor using the ABCDE rule. See a dermatologist if any changes occur.",
        "do_list": [
            "Perform monthly self-skin checks using the ABCDE rule",
            "Get an annual full-body skin exam from a dermatologist",
            "Apply sunscreen daily to prevent UV-induced changes",
            "Keep a photo record of moles to track changes over time",
            "See a doctor immediately if a mole changes shape, color, or size"
        ],
        "dont_list": [
            "Do not pick or scratch at the mole",
            "Avoid excessive sun exposure without protection",
            "Do not use tanning beds",
            "Do not ignore any new moles appearing after age 30",
            "Avoid harsh skin treatments near the mole"
        ]
    },
    "Vascular Lesion": {
        "severity": "Moderate",
        "description": "Abnormalities of blood vessels in or near the skin, including hemangiomas, spider veins, and port-wine stains. Most are benign but can bleed or grow. They range from tiny red dots to large port-wine birthmarks.",
        "recommendation": "Consult a doctor if it bleeds, grows rapidly, or causes discomfort. Laser treatment is often effective.",
        "do_list": [
            "Consult a dermatologist or vascular specialist for evaluation",
            "Apply gentle pressure if it bleeds and seek medical help",
            "Ask about laser therapy if it causes cosmetic concern",
            "Apply sunscreen to prevent worsening from UV exposure",
            "Monitor for any increase in size or bleeding frequency"
        ],
        "dont_list": [
            "Do not scratch or pick at the lesion",
            "Avoid blood-thinning medications without doctor approval",
            "Do not ignore rapid growth or frequent bleeding",
            "Avoid high-impact trauma to the affected area",
            "Do not use home remedies to try to remove it"
        ]
    },
    "Unknown": {
        "severity": "Unknown",
        "description": "The AI could not confidently identify the condition from the uploaded image. This may be due to image quality, lighting, or an uncommon presentation.",
        "recommendation": "Please provide a clearer, well-lit photo of the affected area or consult a medical professional.",
        "do_list": [
            "Take a clearer, well-lit photo and try again",
            "Consult a licensed dermatologist for professional diagnosis",
            "Describe any symptoms (itching, pain, growth) to your doctor",
            "Keep the area clean and avoid touching it unnecessarily"
        ],
        "dont_list": [
            "Do not self-diagnose or self-treat based on this result",
            "Do not ignore the condition if symptoms persist",
            "Avoid applying random creams or treatments without guidance"
        ]
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
        "recommendation": info["recommendation"],
        "do_list": info.get("do_list", []),
        "dont_list": info.get("dont_list", [])
    }
