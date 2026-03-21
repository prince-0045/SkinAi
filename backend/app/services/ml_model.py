"""
Skin Disease Prediction Service using MobileNetV2.
13-class model (pre-merged during training).
"""

import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
_H5_PATH = BASE_DIR / "model.weights.h5"

_PKL_PATH = None
IMAGE_SIZE = 224
NUM_CLASSES = 13

# 13 final class labels (pre-merged during training)
CLASS_NAMES = [
    'Acne', 'Actinic_Keratosis', 'Benign_Growth', 'DrugEruption', 'Eczema',
    'Fungal_Infection', 'Infestations_Bites', 'Psoriasis', 'Rosacea',
    'SkinCancer', 'Unknown', 'Vitiligo', 'Warts'
]

# ── No merging needed — classes are already merged at training time ──
# Only map Unknown_Normal → Unknown if the model ever outputs it
CLASS_MERGE_MAP = {}

# Sub-conditions shown in the UI for merged classes
MERGED_CLASS_INCLUDES = {
    'Fungal_Infection': ['Tinea (Ringworm)', 'Candidiasis'],
    'Benign_Growth':    ['Moles', 'Seborrheic Keratosis', 'Benign Tumors'],
}

# ── Disease Information for all 13 final UI categories ──
DISEASE_INFO = {
    "Acne": {
        "severity": "Mild",
        "description": "A common skin condition where hair follicles become clogged with oil and dead skin cells, leading to pimples, blackheads, or whiteheads. It most often appears on the face, forehead, chest, upper back, and shoulders.",
        "recommendation": "Use a gentle cleanser and consider over-the-counter treatments. Consult a dermatologist if acne is severe or persistent.",
        "do_list": [
            "Wash affected areas gently twice daily",
            "Use non-comedogenic skincare and makeup products",
            "Apply prescribed topical treatments consistently",
            "Keep hands away from the face",
            "Stay hydrated and maintain a balanced diet"
        ],
        "dont_list": [
            "Do not pop, squeeze, or pick at pimples",
            "Avoid harsh scrubbing or over-washing",
            "Do not use oily or greasy cosmetics",
            "Avoid touching your face frequently",
            "Do not skip moisturizer — even oily skin needs hydration"
        ]
    },
    "Actinic_Keratosis": {
        "severity": "Moderate",
        "description": "A precancerous rough, scaly patch on the skin caused by years of sun exposure. It appears as a dry, flaky patch that may itch or burn. Early treatment is important to prevent progression to skin cancer.",
        "recommendation": "See a dermatologist for evaluation and possible cryotherapy, topical creams, or photodynamic therapy.",
        "do_list": [
            "Apply broad-spectrum SPF 30+ sunscreen daily",
            "Wear protective clothing and wide-brimmed hats outdoors",
            "Visit a dermatologist for professional evaluation",
            "Perform regular self-skin checks for new patches",
            "Keep affected skin moisturized"
        ],
        "dont_list": [
            "Do not scratch or pick at the rough patches",
            "Avoid prolonged sun exposure especially 10am–4pm",
            "Do not use tanning beds or UV lamps",
            "Do not ignore it — untreated it can progress to skin cancer",
            "Avoid harsh exfoliants on affected areas"
        ]
    },
    "DrugEruption": {
        "severity": "Moderate",
        "description": "An adverse skin reaction caused by certain medications. It can present as rashes, hives, blisters, or red patches, and may appear days to weeks after starting a new medication.",
        "recommendation": "Stop the suspected medication (after consulting your doctor) and seek medical attention immediately.",
        "do_list": [
            "Note the exact medications taken before the reaction",
            "Consult a doctor immediately about the reaction",
            "Keep a record of all drug allergies",
            "Apply soothing lotions for relief"
        ],
        "dont_list": [
            "Do not stop medication without consulting a doctor first",
            "Avoid retaking the suspected medication",
            "Do not ignore spreading rashes or fever",
            "Do not apply strong topical steroids without prescription"
        ]
    },
    "Eczema": {
        "severity": "Moderate",
        "description": "A chronic condition that makes skin red, itchy, and inflamed. It often appears in patches on the hands, feet, ankles, wrists, neck, upper chest, eyelids, and inside the bend of elbows and knees.",
        "recommendation": "Use fragrance-free moisturizers regularly and consult a dermatologist for prescription treatments if needed.",
        "do_list": [
            "Moisturize skin at least twice daily with fragrance-free products",
            "Use gentle, soap-free cleansers",
            "Wear soft, breathable fabrics like cotton",
            "Identify and avoid personal triggers",
            "Apply prescribed topical medications consistently"
        ],
        "dont_list": [
            "Do not scratch — it worsens inflammation",
            "Avoid hot showers — use lukewarm water",
            "Do not use fragranced soaps or detergents",
            "Avoid wool or rough fabrics against the skin",
            "Do not skip moisturizing even when skin looks clear"
        ]
    },
    "Fungal_Infection": {
        "severity": "Mild",
        "description": "A group of skin conditions caused by fungal organisms, including ringworm (tinea) and candidiasis. They typically cause red, itchy, scaly, or ring-shaped patches on the skin.",
        "recommendation": "Use antifungal creams or medications as directed. Consult a doctor if the infection does not improve within two weeks.",
        "do_list": [
            "Keep affected areas clean and dry",
            "Use prescribed antifungal treatments consistently",
            "Wash towels, clothes, and bedding frequently",
            "Wear loose-fitting, breathable clothing",
            "Complete the full course of antifungal treatment"
        ],
        "dont_list": [
            "Do not share towels, clothing, or personal items",
            "Avoid scratching the affected area",
            "Do not stop treatment early even if symptoms improve",
            "Avoid tight, non-breathable clothing",
            "Do not walk barefoot in public showers or pools"
        ]
    },
    "Infestations_Bites": {
        "severity": "Mild",
        "description": "Skin reactions caused by insect bites or parasitic infestations such as scabies, lice, or bed bugs. They typically present as itchy bumps, rashes, or tracks on the skin.",
        "recommendation": "Identify the source and use appropriate anti-itch or anti-parasitic treatments. Consult a doctor for persistent infestations.",
        "do_list": [
            "Clean the affected area with soap and water",
            "Apply anti-itch creams or cold compress for relief",
            "Wash all clothing and bedding in hot water",
            "Consult a doctor for prescription treatment if needed"
        ],
        "dont_list": [
            "Do not scratch — it can cause secondary infection",
            "Avoid sharing personal items during infestation",
            "Do not ignore persistent itching lasting days",
            "Avoid self-treating scabies without medical confirmation"
        ]
    },
    "Psoriasis": {
        "severity": "Moderate",
        "description": "A chronic autoimmune condition that speeds up skin cell production, causing thick, red, scaly patches that can be itchy and sometimes painful. Commonly affects elbows, knees, scalp, and lower back.",
        "recommendation": "Consult a dermatologist for topical treatments, light therapy, or systemic medications. Moisturize regularly.",
        "do_list": [
            "Moisturize skin daily to reduce dryness and scaling",
            "Use prescribed topical treatments consistently",
            "Consider phototherapy under doctor supervision",
            "Track and avoid personal triggers (stress, cold, etc.)",
            "Maintain a healthy weight and balanced diet"
        ],
        "dont_list": [
            "Do not scratch or pick at scales",
            "Avoid alcohol and smoking — both worsen psoriasis",
            "Do not use harsh soaps or hot water",
            "Avoid stress when possible",
            "Do not stop prescribed treatments without consulting your doctor"
        ]
    },
    "Rosacea": {
        "severity": "Mild",
        "description": "A chronic skin condition causing redness, flushing, visible blood vessels, and sometimes small, pus-filled bumps on the face. It commonly affects the cheeks, nose, chin, and forehead.",
        "recommendation": "Identify and avoid triggers. Consult a dermatologist for prescription treatments like topical or oral antibiotics.",
        "do_list": [
            "Use gentle, fragrance-free skincare products",
            "Apply broad-spectrum sunscreen daily",
            "Identify and avoid personal triggers (heat, spicy food, alcohol)",
            "Use green-tinted primer to neutralize redness if desired"
        ],
        "dont_list": [
            "Avoid hot beverages and spicy foods if they trigger flushing",
            "Do not use harsh exfoliants or astringents",
            "Avoid alcohol-based skincare products",
            "Do not use topical steroids unless prescribed",
            "Avoid extreme temperature changes"
        ]
    },
    "Benign_Growth": {
        "severity": "Low",
        "description": "Non-cancerous skin growths including common moles, seborrheic keratoses, and benign tumors. These are usually harmless but should be monitored for any changes.",
        "recommendation": "Usually no treatment needed. Monitor for changes in size, shape, or color. Consult a doctor if concerned.",
        "do_list": [
            "Monitor growths regularly for any changes",
            "Get an annual skin check from a dermatologist",
            "Apply sunscreen to protect existing growths",
            "Keep a photo record to track changes over time",
            "Consult a doctor if a growth changes shape or bleeds"
        ],
        "dont_list": [
            "Do not pick, scratch, or try to remove growths yourself",
            "Do not ignore sudden changes in appearance",
            "Avoid excessive UV exposure",
            "Do not assume all new growths are benign — get checked",
            "Do not use over-the-counter removers without medical advice"
        ]
    },
    "SkinCancer": {
        "severity": "Critical",
        "description": "A serious condition involving abnormal growth of skin cells, often developing on sun-exposed areas. It can present as unusual moles, changing spots, or non-healing sores. Early detection is critical.",
        "recommendation": "⚠️ IMMEDIATE medical attention required. Contact a dermatologist or oncologist as soon as possible.",
        "do_list": [
            "Seek immediate medical attention — do not wait",
            "Get a biopsy performed by a professional",
            "Document and photograph the lesion",
            "Follow your doctor's treatment plan",
            "Apply extreme sun protection daily",
            "Notify family members — there may be hereditary risk"
        ],
        "dont_list": [
            "Do NOT delay seeking medical care — every day matters",
            "Do not expose the lesion to any sunlight or UV radiation",
            "Do not attempt any home remedies or self-treatment",
            "Do not ignore it or assume it will go away",
            "Avoid tanning beds permanently",
            "Do not miss any follow-up appointments"
        ]
    },
    "Vitiligo": {
        "severity": "Low",
        "description": "A condition where the skin loses its pigment cells (melanocytes), resulting in discolored patches. It is not contagious or life-threatening, but can significantly affect self-esteem.",
        "recommendation": "Consult a dermatologist for treatment options including topical corticosteroids, light therapy, or cosmetic solutions.",
        "do_list": [
            "Protect depigmented areas with SPF 50+ sunscreen",
            "Consult a dermatologist about treatment options",
            "Consider cosmetic camouflage products if desired",
            "Seek emotional support if it affects mental health"
        ],
        "dont_list": [
            "Do not expose affected areas to excessive sun",
            "Avoid tattoos on affected areas (can worsen condition)",
            "Do not use unproven home remedies",
            "Do not ignore the emotional impact — seek support"
        ]
    },
    "Warts": {
        "severity": "Low",
        "description": "Small, rough skin growths caused by the human papillomavirus (HPV). They are contagious and commonly appear on hands, feet, and other areas.",
        "recommendation": "Many warts resolve on their own. Over-the-counter treatments or medical procedures can speed removal.",
        "do_list": [
            "Keep warts covered to prevent spreading",
            "Wash hands thoroughly after touching warts",
            "Use over-the-counter salicylic acid treatments",
            "Consult a doctor for stubborn or painful warts"
        ],
        "dont_list": [
            "Do not pick, scratch, or bite warts",
            "Do not share towels or personal items",
            "Avoid walking barefoot in communal areas",
            "Do not try to cut warts off yourself"
        ]
    },
    "Unknown": {
        "severity": "Unknown",
        "description": "The AI could not identify a skin condition from the uploaded image. This may be because the image is not of a skin condition, or due to image quality issues.",
        "recommendation": "Please upload a clearer, well-lit photo of the affected area, or consult a medical professional directly.",
        "do_list": [
            "Take a clearer, well-lit photo and try again",
            "Ensure good lighting and sharp focus",
            "Consult a licensed dermatologist for professional diagnosis",
            "Describe any symptoms to your doctor"
        ],
        "dont_list": [
            "Do not self-diagnose or self-treat based on this result",
            "Do not ignore the condition if symptoms persist",
            "Avoid applying treatments without professional guidance"
        ]
    }
}

# ── High-risk classes that get red UI treatment ──
HIGH_RISK_CLASSES = {'SkinCancer', 'Actinic_Keratosis'}

_model = None

def _get_model():
    """Load the MobileNetV2 model lazily so the server starts fast."""
    global _model
    if _model is None:
        print(f"[ML] Rebuilding {NUM_CLASSES}-class MobileNetV2 model and loading weights from {_H5_PATH} ...")
        
        # 1. Patch for Windows DLL Application Control policy block
        import sys
        import types
        try:
            mock_mod = types.ModuleType('tensorflow.python._pywrap_quantize_training')
            mock_mod.DoQuantizeTrainingOnGraphDefHelper = lambda *args, **kwargs: None
            sys.modules['tensorflow.python._pywrap_quantize_training'] = mock_mod
        except Exception:
            pass

        # 2. Import modern Keras 3 
        import tensorflow as tf
        import keras
        
        # 3. Rebuild MobileNetV2 architecture exactly as in training script
        print("[ML] Constructing MobileNetV2 backbone...")
        inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")

        # The preprocessing cast and lambda
        x = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(inputs)
        x = keras.layers.Lambda(
            keras.applications.mobilenet_v2.preprocess_input
        )(x)

        base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            pooling='avg'
        )
        base_model._name = "functional"  # Force match the H5 name
        x = base_model(x, training=False)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(512, activation="relu")(x)
        x = keras.layers.Dropout(0.35)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.175)(x)
        outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        
        _model = keras.Model(inputs=inputs, outputs=outputs)
        
        # 4. Load weights natively
        print("[ML] Loading weights...")
        _model.load_weights(str(_H5_PATH))
        
        print(f"[ML] {NUM_CLASSES}-Class MobileNetV2 Model rebuilt and weights loaded successfully!")
    return _model


def predict(image_bytes: bytes):
    """Run real inference on skin image. Returns merged category + metadata."""
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import keras

    # 1. Get the model
    model = _get_model()

    # 2. Preprocess Image
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing is handled internally by the Model's Lambda layer

    # 3. Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    raw_label = CLASS_NAMES[pred_idx]
    confidence = float(predictions[0][pred_idx])
    
    # 4. Apply class merging
    category = CLASS_MERGE_MAP.get(raw_label, raw_label)
    
    # If confidence is exceptionally low, fallback to Unknown
    if confidence < 0.35:
        category = "Unknown"

    is_unknown = (category == "Unknown")
    
    # 5. Get includes (sub-conditions for merged classes)
    includes = MERGED_CLASS_INCLUDES.get(category, [])
    
    # 6. Get detailed info
    info = DISEASE_INFO.get(category, DISEASE_INFO["Unknown"])

    return {
        "category": category,
        "disease": category,  # backward compat
        "confidence": confidence,
        "is_unknown": is_unknown,
        "includes": includes,
        "severity": info["severity"],
        "description": info["description"],
        "recommendation": info["recommendation"],
        "do_list": info.get("do_list", []),
        "dont_list": info.get("dont_list", []),
    }
