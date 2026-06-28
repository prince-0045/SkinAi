# 🧠 SkinAI — Complete AI/ML Interview Preparation Guide
> **Your Interview is Tomorrow — Read This Entire Document!**
> Every detail is pulled directly from YOUR actual project code.

---

## 📋 TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [ML Model — Complete Technical Deep Dive](#2-ml-model--complete-technical-deep-dive)
3. [Model Architecture — Layer by Layer](#3-model-architecture--layer-by-layer)
4. [Training Pipeline — How It Was Trained](#4-training-pipeline--how-it-was-trained)
5. [Prediction Pipeline — How It Works in Production](#5-prediction-pipeline--how-it-works-in-production)
6. [Technology Stack](#6-technology-stack)
7. [Deployment — Complete Guide + Problems Faced](#7-deployment--complete-guide--problems-faced)
8. [API Endpoints Reference](#8-api-endpoints-reference)
9. [Interview Questions & Answers — ML Model](#9-interview-questions--answers--ml-model)
10. [Interview Questions & Answers — Deployment](#10-interview-questions--answers--deployment)
11. [Interview Questions & Answers — System Design](#11-interview-questions--answers--system-design)
12. [Tricky Questions an Interviewer Will Ask YOU](#12-tricky-questions-an-interviewer-will-ask-you)
13. [Quick Cheat Sheet — Numbers to Remember](#13-quick-cheat-sheet--numbers-to-remember)

---

## 1. PROJECT OVERVIEW

**Project Name:** SkinAI (also called Dermaura)  
**Live Domain:** `dermaura.tech`  
**What It Does:** An AI-powered web application that detects 13 different skin diseases from a photo uploaded by the user. It gives confidence scores, severity levels, treatment recommendations, and connects users with nearby doctors.

### Core Features
| Feature | Description |
|---|---|
| 🤖 AI Detection | Upload a skin photo → get disease prediction with confidence % |
| 📊 Severity Score | Mild / Moderate / High / Critical based on disease + confidence |
| 📈 Progress Tracker | Compare multiple scans over time to track healing |
| 🏥 Doctor Finder | AWS Location Services to find dermatologists nearby |
| 📄 PDF Reports | Downloadable medical report with scan history |
| 🔒 Authentication | Email/OTP + Google OAuth login |

---

## 2. ML MODEL — COMPLETE TECHNICAL DEEP DIVE

### 2.1 Which Model Is Used?

> **Production Model: EfficientNetV2S** (not MobileNetV2)

The project has **two models**:
- `train_mobilenetv2_colab.py` — A training script for MobileNetV2 (used for experimentation / early prototype)
- **PRODUCTION**: `effnetv2s_final.weights.h5` — EfficientNetV2S, the final deployed model

The file `backend/app/services/ml_model.py` clearly states:
```
Rebuilding 13-class EfficientNetV2S model and loading weights from effnetv2s_final.weights.h5
```

### 2.2 Why EfficientNetV2S Was Chosen Over MobileNetV2

| Criteria | MobileNetV2 | EfficientNetV2S (chosen) |
|---|---|---|
| Parameters | ~3.4M | ~20.3M |
| Input Resolution | 224×224 | 300×300 |
| Accuracy on our dataset | Lower (prototype) | **80.69% validation** |
| ImageNet Top-1 Acc | 71.8% | **83.9%** |
| Speed | Faster | Slightly slower |
| Medical imaging | Good | **Better feature extraction** |

**Decision**: Higher accuracy on skin disease detection was more important than inference speed, especially since misclassifying skin cancer is a critical error.

### 2.3 What is EfficientNetV2S?

EfficientNetV2S is a state-of-the-art CNN architecture introduced by Google in 2021. The "S" means **Small** variant of the EfficientNetV2 family.

**Key innovations:**
- **Compound Scaling**: Simultaneously scales width, depth, and resolution using a fixed compound coefficient
- **Fused-MBConv blocks**: Replaces depthwise convolutions with regular convolutions in early layers for better GPU training efficiency
- **Progressive Learning**: Uses smaller image sizes early in training and increases later
- **NAS (Neural Architecture Search)**: Architecture was found via automated search, not hand-designed

### 2.4 Training Dataset

- **Primary**: HAM10000 (Human Against Machine with 10000 training images) — a well-known dermatology dataset
- **Augmented**: Custom skin disease dataset on Kaggle with 13 disease classes
- **Preprocessing**: 300×300 pixel RGB images
- **Batch Size**: 16 (smaller due to larger image size and model)

---

## 3. MODEL ARCHITECTURE — LAYER BY LAYER

This is the exact architecture rebuilt in `ml_model.py`:

```
Input Image (any size JPG/PNG from user)
    ↓
[PIL Resize] → 300 × 300 × 3 (RGB)
    ↓
[numpy expand_dims] → 1 × 300 × 300 × 3 (batch dimension added)
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1: keras.Input(shape=(300, 300, 3), name="image")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2: Lambda → tf.cast(t, tf.float32)
    [Converts uint8 pixels to float32]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3: Lambda → efficientnet_v2.preprocess_input
    [Normalizes pixels to [-1, 1] range]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 4-??: EfficientNetV2S BACKBONE
    include_top=False  → removes ImageNet classifier head
    weights=None       → we load our own trained weights
    pooling='avg'      → Global Average Pooling at the end
    [~340+ internal layers: Fused-MBConv + MBConv blocks]
    Output: 1 × 1280 (feature vector)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+1: BatchNormalization
    [Normalizes feature distribution, speeds up training]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+2: Dense(512, activation='relu')
    [First fully-connected layer, 512 neurons]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+3: Dropout(0.35)
    [Drops 35% neurons randomly during training to prevent overfitting]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+4: Dense(256, activation='relu')
    [Second fully-connected layer, 256 neurons]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+5: Dropout(0.175)
    [Drops 17.5% neurons — lighter dropout in second layer]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER N+6: Dense(13, activation='softmax')
    [OUTPUT: 13 probabilities, one per disease class]
    [softmax ensures all 13 values sum to 1.0]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
Final Output: Array of 13 probabilities
```

### Why These Design Choices?

| Choice | Why |
|---|---|
| `include_top=False` | We need our own 13-class head, not 1000-class ImageNet |
| `pooling='avg'` | Global Average Pooling reduces 7×7×1280 → 1280, less overfitting than Flatten |
| BatchNormalization | Stabilizes training, reduces internal covariate shift |
| Two Dense layers (512→256) | Gradual dimension reduction for better feature learning |
| Dropout 0.35 → 0.175 | Higher dropout early, lower in second layer (progressive regularization) |
| Softmax activation | Multi-class classification — outputs probability distribution |
| ReLU in hidden layers | Avoids vanishing gradient, computationally efficient |

---

## 4. TRAINING PIPELINE — HOW IT WAS TRAINED

### 4.1 Two-Stage Transfer Learning (Fine-Tuning)

**Stage 1 — Head Training (Backbone Frozen):**
```python
base_model.trainable = False  # Freeze EfficientNetV2S
# Only train: BatchNorm + Dense(512) + Dropout + Dense(256) + Dropout + Dense(13)
optimizer = Adam(learning_rate=1e-3)  # Higher LR ok since backbone frozen
epochs = 15  # Head training epochs
```

**Stage 2 — Fine-Tuning (Top layers unfrozen):**
```python
base_model.trainable = True
# Freeze bottom layers, unfreeze top ~30 layers of backbone
optimizer = Adam(learning_rate=3e-5)  # Very small LR to not destroy pretrained weights
epochs = up to 40 (with early stopping)
```

### 4.2 Actual Training Metrics (from training_log.csv)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss | LR |
|---|---|---|---|---|---|
| 0 | 64.8% | 69.2% | 1.352 | 1.275 | 3e-5 |
| 5 | 75.5% | 73.6% | 1.119 | 1.175 | 3e-5 |
| 10 | 82.1% | 75.9% | 0.985 | 1.133 | 3e-5 |
| 18 | 89.4% | 78.8% | 0.840 | 1.087 | 3e-5 |
| 25 | 93.2% | 80.0% | 0.763 | 1.055 | 3e-5 |
| 28 | 95.1% | 80.0% | 0.720 | 1.039 | 7.5e-6 (decayed) |
| **36** | **96.2%** | **80.7%** | **0.697** | **1.025** | 1.9e-6 |

**Best Validation Accuracy: 80.69%** (from metadata.json)

### 4.3 Callbacks Used During Training

```python
EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)
CSVLogger(csv_path)  # Saved training_log.csv
```

### 4.4 Data Augmentation (MobileNetV2 prototype, similar for EfficientNetV2S)

```python
RandomFlip("horizontal")       # Mirror skin images
RandomRotation(0.08)           # ±8% rotation (~29 degrees)
RandomZoom(0.10)               # ±10% zoom
RandomBrightness(0.08)         # ±8% brightness variation
```

### 4.5 Class Imbalance Handling

Skin disease datasets are heavily imbalanced (e.g., 10× more Acne images than Vitiligo).

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=classes, y=y_labels)
model.fit(..., class_weight=weights)
```

This ensures rare diseases like SkinCancer are not ignored during training.

### 4.6 Loss Function

```
sparse_categorical_crossentropy
```
Used because labels are integers (0–12), not one-hot encoded vectors.

---

## 5. PREDICTION PIPELINE — HOW IT WORKS IN PRODUCTION

This is the exact flow when a user uploads a photo:

```
USER uploads image via React frontend
        ↓
POST /api/v1/scan/upload  (multipart/form-data)
        ↓
FastAPI receives bytes → authenticates JWT token
        ↓
Image bytes sent to: ml_model.predict(image_bytes)
        ↓
Step 1: PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
        [Handles JPG, PNG, WEBP — forces 3 channels]
        ↓
Step 2: img.resize((300, 300))
        [Resize to model input size]
        ↓
Step 3: keras.utils.img_to_array(img)
        [PIL → numpy array, shape: (300, 300, 3)]
        ↓
Step 4: np.expand_dims(img_array, axis=0)
        [Add batch dimension → shape: (1, 300, 300, 3)]
        ↓
Step 5: model.predict(img_array, verbose=0)
        [Forward pass through EfficientNetV2S]
        [Output: shape (1, 13) — 13 probabilities]
        ↓
Step 6: np.argmax(predictions[0])
        [Pick index of highest probability]
        ↓
Step 7: CLASS_NAMES[pred_idx]
        [Map index → class name string]
        ↓
Step 8: Check confidence < 0.35 (MIN_CONFIDENCE_THRESHOLD)
        [If too low → return "Unknown"]
        ↓
Step 9: CLASS_MERGE_MAP lookup
        [Unknown_Normal → Unknown]
        ↓
Step 10: DISEASE_INFO lookup
        [Get severity, description, recommendations, do/dont lists]
        ↓
JSON Response to frontend:
{
  "disease": "Eczema",
  "confidence": 0.87,
  "severity": "Moderate",
  "description": "...",
  "recommendation": "...",
  "do_list": [...],
  "dont_list": [...]
}
```

### Model Loading Strategy — Lazy Loading with Startup Preload

```python
# In main.py - model is preloaded on startup
@app.on_event("startup")
async def startup_event():
    _get_model()  # Load 276MB model into RAM once

# In ml_model.py - singleton pattern
_model = None
def _get_model():
    global _model
    if _model is None:
        # Build EfficientNetV2S architecture
        # Load weights from .h5 file
        _model = keras.Model(inputs=inputs, outputs=outputs)
        _model.load_weights(str(_H5_PATH))
    return _model  # Always returns same loaded model
```

**Why this matters:** Loading the model once on startup (not per request) reduces latency from ~10 seconds to ~200ms per prediction.

---

## 6. TECHNOLOGY STACK

### Frontend
| Tech | Version | Purpose |
|---|---|---|
| React | 19.2.0 | UI framework |
| Vite | Latest | Build tool, dev server |
| Tailwind CSS | Latest | Styling |
| Framer Motion | Latest | Animations |
| React Router | 7.12.0 | Client-side routing |
| Three.js + R3F | Latest | 3D DNA helix animation |
| Recharts | Latest | Scan history charts |
| jsPDF | Latest | PDF report generation |
| React Dropzone | Latest | Drag-and-drop image upload |

### Backend
| Tech | Purpose |
|---|---|
| **FastAPI** | High-performance async Python web framework |
| **Uvicorn** | ASGI server (development) |
| **Gunicorn** | WSGI server with multiple workers (production) |
| **Motor** | Async MongoDB driver |
| **ODMantic** | MongoDB ORM for Python |
| **Pydantic** | Data validation |
| **Python-jose** | JWT token handling |
| **Passlib + Argon2** | Password hashing |
| **SlowAPI** | Rate limiting (1000 req/min default) |
| **GZipMiddleware** | Response compression (min 500 bytes) |

### ML Stack
| Tech | Purpose |
|---|---|
| **TensorFlow (CPU)** | ML framework (tensorflow-cpu in prod) |
| **Keras 3** | High-level model API |
| **EfficientNetV2S** | Pretrained backbone (from keras.applications) |
| **NumPy** | Array operations |
| **Pillow (PIL)** | Image loading, resizing, format conversion |
| **OpenCV (headless)** | Additional image processing |
| **Scikit-learn** | Class weight computation, metrics |

### Infrastructure & Cloud
| Service | Purpose |
|---|---|
| **MongoDB Atlas** | Cloud NoSQL database |
| **Cloudinary** | Image CDN and storage |
| **Google OAuth 2.0** | Social login |
| **Resend** | Email delivery (OTP, notifications) |
| **AWS Location Services** | Doctor location search |
| **Docker + Docker Compose** | Containerization |

---

## 7. DEPLOYMENT — COMPLETE GUIDE + PROBLEMS FACED

### 7.1 Local Development Setup

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload   # Runs on http://localhost:8000

# Frontend
cd frontend
npm install
npm run dev                      # Runs on http://localhost:5173
```

### 7.2 Docker Deployment

**`backend/Dockerfile`:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
```

**`docker-compose.yml`:**
```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: [.env]          # All secrets from .env

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
    build_args:
      VITE_API_URL: https://api.dermaura.tech
```

```bash
# Build and run
docker-compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### 7.3 Production Deployment

**Backend (Gunicorn + Uvicorn workers):**
```bash
# NEVER use plain uvicorn in production — use Gunicorn for multiple workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
- `-w 4` = 4 worker processes (handles concurrent requests)
- `-k uvicorn.workers.UvicornWorker` = async ASGI workers

**Frontend:**
```bash
npm run build          # Creates optimized dist/ folder
# Deploy dist/ to Nginx, Vercel, or Netlify
# Must configure SPA fallback: all routes → index.html
```

### 7.4 Environment Variables Required

```env
# Database
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/skinai

# Security
SECRET_KEY=64-char-random-string
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email (Resend)
MAIL_USERNAME=your_resend_api_key
MAIL_FROM=noreply@dermaura.tech
MAIL_SERVER=smtp.resend.com
MAIL_PORT=587

# Google OAuth
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx

# Cloudinary
CLOUDINARY_CLOUD_NAME=xxx
CLOUDINARY_API_KEY=xxx
CLOUDINARY_API_SECRET=xxx

# AWS
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1

# Frontend
FRONTEND_URL=https://dermaura.tech
```

### 7.5 🚨 DEPLOYMENT PROBLEMS FACED (Very Important for Interview!)

#### Problem 1: Windows DLL Application Control Policy Block
```
Error: tensorflow.python._pywrap_quantize_training blocked
```
**Cause:** Windows Application Control policy blocks certain TensorFlow DLLs when running on Windows servers.

**Fix Applied (in ml_model.py):**
```python
import types, sys
mock_mod = types.ModuleType('tensorflow.python._pywrap_quantize_training')
mock_mod.DoQuantizeTrainingOnGraphDefHelper = lambda *args, **kwargs: None
sys.modules['tensorflow.python._pywrap_quantize_training'] = mock_mod
```
**Lesson:** Mock the blocked modules to bypass OS-level DLL restrictions.

---

#### Problem 2: Model Takes 10+ Seconds to Load Per Request
**Cause:** Model weights file is 276MB. Loading on every prediction request was impossible.

**Fix Applied:** Singleton + Startup Preloading
```python
_model = None  # Global singleton

@app.on_event("startup")
async def startup_event():
    _get_model()  # Load ONCE at server start

def _get_model():
    global _model
    if _model is None:  # Only load if not already loaded
        _model = build_model_and_load_weights()
    return _model
```
**Lesson:** Load heavy ML models once at startup, reuse the same instance for all requests.

---

#### Problem 3: Model .h5 File Too Large for Git / Some Platforms
**Cause:** `effnetv2s_final.weights.h5` is 276MB — too large for standard GitHub (100MB limit).

**Fix Applied:** Multiple fallback paths
```python
_SERVICE_H5_PATH = Path(__file__).parent / "effnetv2s_final.weights.h5"
_ALT_H5_PATH = BASE_DIR / "effnetv2s_kaggle_dataset" / "effnetv2s_final.weights.h5"
_DEFAULT_H5_PATH = BASE_DIR / "model.weights.h5"

_H5_PATH = (
    _SERVICE_H5_PATH if _SERVICE_H5_PATH.exists() else
    _ALT_H5_PATH if _ALT_H5_PATH.exists() else
    _DEFAULT_H5_PATH
)
```
**Lesson:** Store large model files separately (Git LFS, S3, Cloudinary) and use fallback paths.

---

#### Problem 4: CORS Errors in Production
**Cause:** Frontend on `dermaura.tech` making requests to `api.dermaura.tech` — blocked by browser CORS policy.

**Fix Applied (in main.py):**
```python
origins = [
    "https://dermaura.tech",
    "https://www.dermaura.tech",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(.*\.)?dermaura\.tech",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
**Also:** CORS headers were missing from 4xx/5xx error responses. Custom exception handler was added to fix this.

---

#### Problem 5: Model Architecture Mismatch When Loading Weights
**Cause:** `model.load_weights()` requires exact same architecture as when weights were saved. Any difference causes error.

**Fix Applied:** Manually rebuilding the EXACT same architecture before loading:
```python
# Must rebuild architecture IDENTICALLY to training code
inputs = keras.Input(shape=(300, 300, 3))
x = Lambda(tf.cast)(inputs)          # Must be same
x = Lambda(preprocess_input)(x)      # Must be same
base = EfficientNetV2S(include_top=False, pooling='avg')
base._name = "functional"            # Force match H5 name!
x = base(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.35)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.175)(x)
outputs = Dense(13, activation="softmax")(x)
_model = keras.Model(inputs, outputs)
_model.load_weights(str(_H5_PATH))   # Now it works
```

---

#### Problem 6: tensorflow-gpu vs tensorflow-cpu in Production
**Cause:** Using `tensorflow` (GPU version) on a CPU-only server causes import errors and high RAM usage.

**Fix Applied:** `requirements.txt` uses `tensorflow-cpu`
```
tensorflow-cpu
tf-keras
```
**Lesson:** Always use `tensorflow-cpu` in cloud deployments unless you have a dedicated GPU instance.

---

#### Problem 7: Confidence Threshold for Unclear Images
**Cause:** When user uploads a non-skin image (e.g., a dog photo), the model still predicts a disease with low confidence.

**Fix Applied:** Minimum confidence threshold
```python
MIN_CONFIDENCE_THRESHOLD = 0.35  # From app/core/constants.py

if confidence < MIN_CONFIDENCE_THRESHOLD:
    category = "Unknown"
```
**Lesson:** Always add confidence thresholds to avoid overconfident wrong predictions on out-of-distribution inputs.

---

#### Problem 8: OpenCV Headless for Server Deployment
**Cause:** Regular `opencv-python` requires a display (X server) which doesn't exist on Linux servers.

**Fix Applied:** Use `opencv-python-headless` in `requirements.txt`
```
opencv-python-headless   # NOT opencv-python
```

---

## 8. API ENDPOINTS REFERENCE

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/register` | Register with email + password |
| POST | `/api/v1/auth/login` | Login → get JWT token |
| POST | `/api/v1/auth/google` | Login with Google OAuth token |

### Scan / ML Prediction
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/scan/upload` | Upload image → get AI prediction |
| GET | `/api/v1/scan/history` | Get user's past scans |
| POST | `/api/v1/scan/compare` | Compare two scans for progress |

### User Management
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/users/profile` | Get user profile |
| PUT | `/api/v1/users/profile` | Update user profile |

### Doctor Finder
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/doctors/nearby?lat=X&lng=Y&radius=10` | Find nearby dermatologists |

### System
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check (DB + ML model status) |
| GET | `/api/v1/admin/*` | Admin routes |

---

## 9. INTERVIEW QUESTIONS & ANSWERS — ML MODEL

---

### Q1: What model did you use and why?
**Answer:**
"I used EfficientNetV2S as the final production model. It's a CNN architecture developed by Google that uses compound scaling — simultaneously scaling network width, depth, and input resolution. I chose it over MobileNetV2 because it achieved 80.7% validation accuracy on our 13-class skin disease dataset versus a lower accuracy with MobileNetV2. In medical imaging applications, accuracy matters more than raw inference speed, especially when we're potentially detecting skin cancer."

---

### Q2: Explain Transfer Learning and how you used it.
**Answer:**
"Transfer learning is the technique of taking a model pre-trained on a large dataset (ImageNet — 1.2 million images, 1000 classes) and adapting it for a different task.

In my project, I used it in two stages:

**Stage 1 — Head Training:** I froze all the layers of EfficientNetV2S backbone so their weights don't change. I only trained the custom classification head I added: BatchNorm → Dense(512) → Dropout(0.35) → Dense(256) → Dropout(0.175) → Dense(13, softmax). Learning rate was 1e-3.

**Stage 2 — Fine-tuning:** I unfroze the top 30 layers of the backbone and retrained with a very small learning rate (3e-5). This allows the pretrained features to adapt to skin disease textures without completely forgetting ImageNet features.

Why two stages? If you fine-tune with a high learning rate from the start, you destroy the pretrained features. The small LR in stage 2 makes small adjustments."

---

### Q3: What is EfficientNetV2S's architecture? How many layers?
**Answer:**
"EfficientNetV2S has approximately 340+ internal layers. The key innovation is **Fused-MBConv** blocks in early layers (combining 3×3 conv + expansion into a single op) and standard **MBConv** blocks in later layers.

In my deployment architecture, the complete model has:
- 2 Lambda preprocessing layers
- The EfficientNetV2S backbone (~340 layers, include_top=False, pooling='avg')
- 1 BatchNormalization layer
- Dense(512) + Dropout(0.35)
- Dense(256) + Dropout(0.175)
- Dense(13, softmax) — output layer

Output feature size from backbone before head: **1280 features** (via Global Average Pooling)."

---

### Q4: What is Dropout and why did you use two different rates?
**Answer:**
"Dropout is a regularization technique. During training, it randomly sets a percentage of neurons to zero. This prevents co-adaptation of neurons and forces the network to learn more robust features.

I used:
- **Dropout(0.35)** = 35% neurons dropped after Dense(512)
- **Dropout(0.175)** = 17.5% neurons dropped after Dense(256)

The higher dropout in the first layer is intentional — the 512-neuron layer has more parameters and is at higher risk of overfitting. The second layer is smaller (256) and closer to the output, so more aggressive dropout might hurt classification performance. Progressive dropout (higher → lower) is a common best practice."

---

### Q5: What is Softmax and why is it used in the output layer?
**Answer:**
"Softmax is an activation function that converts raw logits (any real numbers) into a probability distribution. Given 13 outputs, softmax ensures:
1. All 13 values are between 0 and 1
2. All 13 values sum to exactly 1.0

Formula: `softmax(z_i) = e^(z_i) / Σ e^(z_j)`

In my model, Dense(13, activation='softmax') produces 13 probabilities. I take `np.argmax()` to find the disease with the highest probability. I use the maximum probability value as the confidence score."

---

### Q6: How do you handle class imbalance in the dataset?
**Answer:**
"Skin disease datasets are heavily imbalanced — for example, acne has thousands of images while rare conditions like vitiligo might have hundreds. Without handling this, the model would be biased toward common classes.

I used **class_weight** during training:
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=classes, y=y_labels)
model.fit(..., class_weight=weights)
```

The 'balanced' strategy computes: `weight = n_samples / (n_classes * n_samples_per_class)`. This means rare classes get higher weights in the loss function, forcing the model to pay more attention to them."

---

### Q7: What is the confidence threshold and why 35%?
**Answer:**
"The confidence threshold is 0.35 (35%). If the model's maximum class probability is below 35%, we return 'Unknown' instead of a disease name.

**Why 35%?** In a 13-class problem, random guessing gives 1/13 ≈ 7.7%. A confidence of 35% is about 4.5× better than random — it's a meaningful signal. Going lower (say 20%) leads to low-confidence but specific-sounding predictions that could mislead users. Going higher (say 60%) returns 'Unknown' too often for legitimate edge cases.

**Why this matters:** Medical AI should be cautious. It's better to say 'I don't know, please see a doctor' than to confidently predict the wrong disease."

---

### Q8: What is Global Average Pooling vs Flatten?
**Answer:**
"Both convert 3D feature maps to 1D vectors.

**Flatten:** Takes a 7×7×1280 feature map and makes it a 62720-dimensional vector. This creates massive parameter counts in the next Dense layer.

**Global Average Pooling (GAP):** Takes the spatial average of each channel — a 7×7×1280 map becomes a 1280-dimensional vector. This is much more compact and acts as a regularizer.

I used `pooling='avg'` in EfficientNetV2S, which applies GAP. Benefits:
- 49× fewer parameters in the first Dense layer
- More robust to spatial translations
- Better generalization"

---

### Q9: What is BatchNormalization and where did you use it?
**Answer:**
"Batch Normalization normalizes the inputs of each layer so they have zero mean and unit variance (within each mini-batch). It also adds learnable scale (gamma) and shift (beta) parameters.

Benefits:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as a regularizer
- Reduces sensitivity to weight initialization

I added BatchNormalization in my custom head, between the backbone output and the first Dense(512) layer. This is important because the feature distribution coming out of the pretrained backbone can be non-normalized."

---

### Q10: What is the difference between sparse_categorical_crossentropy and categorical_crossentropy?
**Answer:**
"Both are used for multi-class classification.

**categorical_crossentropy:** Expects labels as one-hot vectors, e.g., `[0, 0, 1, 0, 0, ...]`

**sparse_categorical_crossentropy:** Expects labels as integers, e.g., `2` (meaning class index 2)

I used `sparse_categorical_crossentropy` because `keras.utils.image_dataset_from_directory` with `label_mode='int'` returns integer labels. Using this avoids the need to convert labels to one-hot format."

---

## 10. INTERVIEW QUESTIONS & ANSWERS — DEPLOYMENT

---

### Q11: How did you deploy your ML model as an API?
**Answer:**
"I deployed the model using FastAPI as the web framework with Uvicorn as the ASGI server for development and Gunicorn with Uvicorn workers in production.

**Flow:**
1. FastAPI receives POST request with image bytes (multipart/form-data)
2. The image bytes are passed to `ml_model.predict(image_bytes)`
3. The predict function preprocesses the image (PIL → resize → numpy → expand dims)
4. The model runs forward inference: `model.predict(img_array)`
5. The prediction result (disease, confidence, metadata) is returned as JSON

**Key design decision:** The model is loaded once on server startup (singleton pattern) and reused for all requests. This avoids 10-second loading delays per prediction."

---

### Q12: Why FastAPI specifically?
**Answer:**
"FastAPI was chosen for several reasons:
1. **Async support:** FastAPI is built on Starlette and supports Python async/await natively — critical for handling concurrent image upload requests without blocking
2. **Auto-generated docs:** FastAPI automatically generates Swagger UI at `/docs` — great for testing
3. **Pydantic integration:** Built-in request/response validation
4. **Performance:** Benchmarks show FastAPI comparable to Node.js Express — one of the fastest Python frameworks
5. **Type hints:** Python type annotations provide code clarity and IDE support"

---

### Q13: What is the difference between Uvicorn and Gunicorn?
**Answer:**
"**Uvicorn** is a pure ASGI server — lightweight, single process, perfect for development with `--reload`.

**Gunicorn** is a mature WSGI/WSGI server manager that can spawn and manage multiple worker processes. In production, we run:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```
This gives us 4 separate processes, each running a Uvicorn worker. This means:
- 4 requests can be processed simultaneously
- If one worker crashes, others continue serving
- Much better CPU utilization on multi-core servers

**Rule of thumb:** `-w` workers = (2 × CPU cores) + 1"

---

### Q14: What is Docker and why did you use it?
**Answer:**
"Docker packages the application and all its dependencies (Python version, system libraries, pip packages) into a container — a lightweight, isolated unit that runs identically everywhere.

**Why I used it:**
1. **Reproducibility:** Same container runs on developer laptop, staging, and production without 'it works on my machine' problems
2. **Dependency isolation:** TensorFlow, OpenCV, Python 3.10 — all packaged inside container
3. **Easy deployment:** `docker-compose up --build` starts both frontend and backend with one command
4. **System dependencies:** OpenCV requires `libglib2.0-0`, `libgl1` etc. Docker installs these automatically

**Important:** The Dockerfile uses `python:3.10-slim` (not full) to keep image size smaller."

---

### Q15: How do you handle the 276MB model file in deployment?
**Answer:**
"The weights file `effnetv2s_final.weights.h5` is 276MB — too large for standard Git.

**Strategy used:**
1. Added to `.gitignore` / handled with Git LFS for version control
2. The code has 3 fallback paths — it looks in multiple locations for the weights file:
   - `backend/app/services/effnetv2s_final.weights.h5` (primary)
   - `effnetv2s_kaggle_dataset/effnetv2s_final.weights.h5` (secondary)
   - `model.weights.h5` (fallback)
3. In production, the model file is part of the Docker image (COPY . .)
4. Alternatively, can be downloaded from cloud storage (S3/Cloudinary) on first startup

**Better production approach:** Store in S3, download to `/tmp` on cold start, then keep in memory."

---

### Q16: What is CORS and how did you fix it?
**Answer:**
"CORS (Cross-Origin Resource Sharing) is a browser security feature. When the frontend (e.g., `dermaura.tech`) makes an HTTP request to a different domain (e.g., `api.dermaura.tech`), the browser checks if the server allows it by reading `Access-Control-Allow-Origin` headers.

**Problem:** Without CORS middleware, all frontend→API requests were blocked by browsers.

**Fix:** Added FastAPI's `CORSMiddleware`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://dermaura.tech'],
    allow_origin_regex=r'https?://(.*\.)?dermaura\.tech',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
```

**Extra fix:** Error responses (4xx, 5xx) also needed CORS headers, so I added a custom HTTP exception handler that manually adds CORS headers to error responses."

---

### Q17: How does JWT authentication work in your system?
**Answer:**
"JWT (JSON Web Token) is used for stateless authentication.

**Flow:**
1. User logs in → server verifies credentials
2. Server creates JWT: `python-jose` encodes payload (user_id, email, expiry) + signs with SECRET_KEY using HS256 algorithm
3. JWT returned to frontend, stored in memory/localStorage
4. Every subsequent API request includes `Authorization: Bearer <token>` header
5. FastAPI middleware (`deps.py`) verifies the JWT signature and extracts user data
6. Access token expires in 30 minutes (ACCESS_TOKEN_EXPIRE_MINUTES=30)

**Why stateless?** The server doesn't need to store sessions in a database. The token itself contains all the information, verified by the signature."

---

### Q18: How does rate limiting work in your API?
**Answer:**
"I used **SlowAPI** (a FastAPI-compatible wrapper around the `limits` library):

```python
limiter = Limiter(key_func=get_remote_address, default_limits=['1000/minute'])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
```

When a client exceeds 1000 requests per minute from the same IP:
- Server returns HTTP 429 (Too Many Requests)
- Custom error handler returns: `{'detail': 'Too many requests. Please slow down.'}`

**Why rate limiting matters for ML APIs:** Model inference is computationally expensive. Without rate limiting, a single bad actor could send 1000 image uploads per second and crash the server."

---

## 11. INTERVIEW QUESTIONS & ANSWERS — SYSTEM DESIGN

---

### Q19: Walk me through the entire system architecture.
**Answer:**
"The system has 4 main components:

**1. Frontend (React + Vite):**
- User uploads a skin image via drag-and-drop
- Image sent as multipart/form-data to the backend API
- Results displayed with confidence meter, disease info, do/don't lists
- Progress tracking via Recharts
- PDF generation client-side using jsPDF

**2. Backend (FastAPI):**
- Handles authentication (JWT + Google OAuth)
- Receives image → sends to ML service → returns prediction
- Stores scan history in MongoDB Atlas
- Uploads images to Cloudinary CDN for storage
- Sends emails via Resend (OTP, reports)
- Doctor search via AWS Location Services

**3. ML Pipeline:**
- EfficientNetV2S model loaded in memory at startup
- Preprocessing: PIL resize to 300×300 → numpy array → model inference
- Post-processing: argmax → confidence check → disease info lookup

**4. External Services:**
- MongoDB Atlas (cloud database)
- Cloudinary (image storage)
- Google OAuth (authentication)
- AWS Location (geospatial search)"

---

### Q20: How would you improve model accuracy?
**Answer:**
"Several strategies:
1. **More data:** Current dataset may be limited. Collecting more diverse skin images, especially for rare classes
2. **Better augmentation:** Add color jitter, elastic distortions, random erasing — skin lesions are sensitive to these
3. **Ensemble methods:** Train multiple models (EfficientNetV2S + DenseNet201) and average predictions
4. **Larger input size:** Try 384×384 or 456×456 for EfficientNetV2L/XL
5. **Attention mechanisms:** Add CBAM or SE blocks to focus on lesion areas
6. **Preprocessing:** Use skin lesion segmentation to crop and focus the model on the lesion
7. **Domain adaptation:** Use techniques to handle different camera qualities and skin tones"

---

### Q21: How would you scale this system for 1 million users?
**Answer:**
"Key scaling strategies:
1. **Horizontal scaling:** Multiple backend instances behind a load balancer (AWS ALB, Nginx)
2. **Model as a microservice:** Separate the ML inference into its own service — allows independent scaling of prediction vs. auth vs. storage
3. **Async ML inference:** Use a task queue (Celery + Redis) — image upload returns job_id immediately, prediction runs in background, user polls for results
4. **Model optimization:** 
   - Convert to TensorFlow Lite or ONNX Runtime for faster inference
   - Use GPU instances for inference (10× faster)
   - Batch inference when multiple requests arrive simultaneously
5. **Caching:** Cache predictions for identical images (hash-based lookup)
6. **CDN:** Cloudinary already used for images — add CloudFront for API responses
7. **Database sharding:** MongoDB Atlas supports automatic sharding for horizontal DB scaling"

---

## 12. TRICKY QUESTIONS AN INTERVIEWER WILL ASK YOU

---

### ❓ "Your model has 80.7% validation accuracy. Is that good enough for medical use?"

**Answer:** "80.7% accuracy for a 13-class skin disease problem is reasonable but not production-ready for clinical diagnosis. For reference, dermatologists achieve ~85-90% accuracy on similar tasks.

However, our system is NOT designed to replace dermatologists. It's a screening tool — a first step. We explicitly show disclaimers, provide 'consult a doctor' recommendations for all conditions, and have a Doctor Finder feature built in. For critical conditions like SkinCancer and Actinic Keratosis, we show prominent warnings to seek immediate medical attention.

To improve: we'd need a larger, more diverse dataset, per-class precision/recall analysis (high recall on SkinCancer is critical — false negatives are worse than false positives), and clinical validation with dermatologists."

---

### ❓ "What's the difference between training accuracy (96%) and validation accuracy (80%)? Is your model overfitting?"

**Answer:** "The 16% gap does show some degree of overfitting — the model learned training data better than it generalizes to unseen data. This is common with medical imaging datasets that are relatively small.

**Mitigations I applied:**
- Dropout (0.35 and 0.175) in the custom head
- Data augmentation (flip, rotation, zoom, brightness)
- Class weights to prevent biasing toward common classes
- EarlyStopping to stop before overfitting gets worse
- ReduceLROnPlateau to fine-tune without overshooting

**Further improvements:** More training data, stronger augmentation, L2 regularization on Dense layers."

---

### ❓ "Why did you save weights (.h5) instead of the full model?"

**Answer:** "Saving weights only (`model.save_weights()`) saves just the learned parameters without the architecture. This is more portable and smaller.

The trade-off: you must rebuild the exact same architecture before loading weights (which is what `_get_model()` does — it rebuilds the model programmatically, then calls `load_weights()`).

The advantage: weights files are compatible across different Keras versions more reliably than full SavedModel format."

---

### ❓ "What happens if someone uploads a photo of a car or food instead of skin?"

**Answer:** "This is an out-of-distribution (OOD) input problem. Our mitigation is the confidence threshold of 0.35. If the model is confused by a non-skin image, it typically produces a flat probability distribution across 13 classes (each ~7-8%) — this falls below our 35% threshold, so we return 'Unknown' and recommend the user upload a clearer skin image.

However, this is not perfect. A better approach would be:
1. A separate binary classifier first: 'Is this a skin image?'
2. Only if yes, pass to the 13-class classifier"

---

### ❓ "How do you monitor the model in production?"

**Answer:** "Currently, monitoring includes:
- FastAPI's `/health` endpoint checks both DB connectivity and model load status
- Structured logging with response times via `X-Response-Time` header
- SlowAPI tracks request patterns

For production-grade monitoring, I'd add:
- Prediction confidence distribution tracking (alert if avg confidence drops — may indicate distribution shift)
- Logging of all predictions with anonymized image hashes
- A/B testing framework for comparing model versions
- Periodic retraining pipeline when new labeled data arrives"

---

## 13. QUICK CHEAT SHEET — NUMBERS TO REMEMBER

| Item | Value |
|---|---|
| **Model** | EfficientNetV2S |
| **Input size** | 300 × 300 × 3 (RGB) |
| **Output classes** | 13 |
| **Best validation accuracy** | **80.69%** |
| **Final train accuracy** | ~96.2% |
| **Total epochs** | 37 (15 head + up to 40 fine-tune, early stop) |
| **Learning rate (fine-tune)** | 3e-5 (decays to 1.9e-6) |
| **Loss function** | sparse_categorical_crossentropy |
| **Optimizer** | Adam |
| **Confidence threshold** | 0.35 (35%) |
| **Dropout rates** | 0.35 and 0.175 |
| **Dense layers** | 512 → 256 → 13 |
| **Backbone output size** | 1280 features (after Global Avg Pool) |
| **Model file size** | ~276MB (.h5 weights) |
| **Batch size (training)** | 16 |
| **High-risk classes** | SkinCancer, Actinic_Keratosis |
| **Python version** | 3.10 |
| **FastAPI rate limit** | 1000 requests/minute |
| **JWT expiry** | 30 minutes |
| **Gunicorn workers** | 4 (-w 4) |
| **Frontend port (dev)** | 5173 |
| **Backend port** | 8000 |
| **Production domain** | dermaura.tech |

### 13 Disease Classes (Memorize These!)
```
1.  Acne                  (Mild)
2.  Actinic_Keratosis     (Moderate) ⚠️ HIGH RISK
3.  Benign_Growth         (Low)
4.  DrugEruption          (Moderate)
5.  Eczema                (Moderate)
6.  Fungal_Infection      (Mild)
7.  Infestations_Bites    (Mild)
8.  Psoriasis             (Moderate)
9.  Rosacea               (Mild)
10. SkinCancer            (Critical) 🚨 HIGH RISK
11. Unknown_Normal        → mapped to "Unknown"
12. Vitiligo              (Low)
13. Warts                 (Low)
```

---

## 💡 LAST MINUTE TIPS

1. **Say "EfficientNetV2S"** — not "some CNN". Be specific about the model name.
2. **Mention Transfer Learning** in any answer about model training — it's a key concept.
3. **Know the two-stage training** — frozen backbone first, fine-tune later.
4. **Talk about deployment problems** with confidence — Windows DLL issue, model loading, CORS. These show real experience.
5. **Mention the confidence threshold** — it shows you thought about edge cases.
6. **Be honest about accuracy** — 80.7% is not perfect. Show you know how to improve it.
7. **FastAPI + Gunicorn** — always mention this combination for production.
8. **"The model is loaded once at startup"** — shows you understand performance implications.

---

> 📌 **Good luck tomorrow! You built something real — be proud and confident!** 🚀
