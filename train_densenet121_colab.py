"""
DenseNet121 — Skin Disease Classification (Google Colab)
========================================================
Upload this script to Colab, or paste each section into separate cells.

Expected dataset structure on Google Drive (or Colab filesystem):
    SkinDisease/
        train/
            class_a/
            class_b/
            ...
        val/
            class_a/
            class_b/
            ...
        test/
            class_a/
            class_b/
            ...

Outputs (saved to `trained_artifacts/`):
    DenseNet121.h5          — full Keras model
    DenseNet121.pkl         — metadata + pickled model bytes
    DenseNet121_confusion_matrix.png
    DenseNet121_training_curves.png
    artifact_index.json
"""

# ───────────────────────────────── Cell 1: Setup ─────────────────────────────
# %% [markdown]
# # 1. Install dependencies & mount drive

# %%
# !pip install -q tensorflow matplotlib seaborn scikit-learn pillow

# Uncomment these two lines if your dataset lives on Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')

# ───────────────────────────────── Cell 2: Imports ───────────────────────────
# %%
import json
import math
import pickle
import random
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

np.set_printoptions(suppress=True, precision=4)
pd.set_option("display.max_columns", 200)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU devices found:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPUs.")
    except Exception as gpu_err:
        print("Could not set GPU memory growth:", gpu_err)
else:
    print("No GPU detected. Training will run on CPU (slower).")

# ───────────────────────────────── Cell 3: Config ────────────────────────────
# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ──
# If using Google Drive, change PROJECT_ROOT accordingly, e.g.:
#   PROJECT_ROOT = Path("/content/drive/MyDrive/SkinAi")
PROJECT_ROOT = Path.cwd()
TRAIN_DIR = PROJECT_ROOT / "SkinDisease" / "train"
VAL_DIR = PROJECT_ROOT / "SkinDisease" / "val"
TEST_DIR = PROJECT_ROOT / "SkinDisease" / "test"
OUTPUT_DIR = PROJECT_ROOT / "trained_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model-specific hyperparameters (tuned for DenseNet121) ──
MODEL_NAME = "DenseNet121"
IMAGE_SIZE = 224          # DenseNet121 native resolution
BATCH_SIZE = 32           # DenseNet121 is lighter — larger batch OK
DROPOUT_RATE = 0.35       # moderate dropout — DenseNet already has BN
HEAD_EPOCHS = 10          # generous budget; early-stop will cut
FINE_TUNE_EPOCHS = 15     # generous fine-tuning budget
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5       # standard fine-tune LR
FINE_TUNE_LAST_N = 50     # DenseNet121 has ~428 layers; unfreeze top ~50

print("Project root:", PROJECT_ROOT)
print("Train dir:", TRAIN_DIR, "| exists:", TRAIN_DIR.exists())
print("Val dir:", VAL_DIR, "| exists:", VAL_DIR.exists())
print("Test dir:", TEST_DIR, "| exists:", TEST_DIR.exists())
print("Output dir:", OUTPUT_DIR)
print(f"Model: {MODEL_NAME}  |  Image size: {IMAGE_SIZE}  |  Batch: {BATCH_SIZE}")

# ───────────────────────────── Cell 4: Data helpers ──────────────────────────
# %%
def get_class_names(data_dir: Path):
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


def count_images_per_class(data_dir: Path):
    counts = {}
    for cls_name in get_class_names(data_dir):
        cls_dir = data_dir / cls_name
        counts[cls_name] = len([f for f in cls_dir.glob("*") if f.is_file()])
    return counts


def print_data_overview():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Expected SkinDisease/train, SkinDisease/val, and SkinDisease/test directories."
        )
    train_counts = count_images_per_class(TRAIN_DIR)
    val_counts = count_images_per_class(VAL_DIR)
    test_counts = count_images_per_class(TEST_DIR)
    print("Number of train classes:", len(train_counts))
    print("Number of val classes:", len(val_counts))
    print("Number of test classes:", len(test_counts))
    print("-" * 80)
    print("Train class counts:")
    for k, v in train_counts.items():
        print(f"  {k:30s} -> {v}")
    print("-" * 80)
    print("Val class counts:")
    for k, v in val_counts.items():
        print(f"  {k:30s} -> {v}")
    print("-" * 80)
    print("Test class counts:")
    for k, v in test_counts.items():
        print(f"  {k:30s} -> {v}")

    if set(train_counts) != set(test_counts) or set(train_counts) != set(val_counts):
        print("⚠️  Train, Val, and Test class names do NOT match exactly.")
    else:
        print("✅ Train / Val / Test class names match.")


def show_sample_images(data_dir, max_classes=3, samples_per_class=4, img_sz=(128, 128)):
    classes = get_class_names(data_dir)[:max_classes]
    if not classes:
        print("No classes found.")
        return
    plt.figure(figsize=(samples_per_class * 3, max_classes * 3))
    idx = 1
    for cls in classes:
        files = [f for f in (data_dir / cls).glob("*") if f.is_file()][:samples_per_class]
        for fp in files:
            img = Image.open(fp).convert("RGB").resize(img_sz)
            plt.subplot(max_classes, samples_per_class, idx)
            plt.imshow(img)
            plt.title(cls, fontsize=9)
            plt.axis("off")
            idx += 1
    plt.tight_layout()
    plt.show()


print_data_overview()
show_sample_images(TRAIN_DIR)

# ────────────────────────── Cell 5: Dataset creation ─────────────────────────
# %%
def make_datasets(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    print(f"Creating datasets — {image_size}×{image_size}, batch {batch_size}")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        seed=SEED,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        seed=SEED,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    print("Detected class names:", class_names)
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches  :", tf.data.experimental.cardinality(val_ds).numpy())
    print("Test batches :", tf.data.experimental.cardinality(test_ds).numpy())
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights_from_train_dir(class_names):
    print("Computing class weights …")
    y = []
    for ci, cn in enumerate(class_names):
        files = [f for f in (TRAIN_DIR / cn).glob("*") if f.is_file()]
        y.extend([ci] * len(files))
    classes = np.arange(len(class_names))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(y))
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    for ci, w in cw.items():
        print(f"  class {ci:2d} ({class_names[ci]}): {w:.4f}")
    return cw


train_ds, val_ds, test_ds, class_names = make_datasets()
class_weights = compute_class_weights_from_train_dir(class_names)
NUM_CLASSES = len(class_names)
print(f"\nTotal classes: {NUM_CLASSES}")

# ─────────────────────────── Cell 6: Build model ─────────────────────────────
# %%
def build_densenet121(num_classes: int):
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    print(f"[{MODEL_NAME}] Building model — input {input_shape}")

    inputs = keras.Input(shape=input_shape, name="image")

    # Data augmentation (runs on GPU in training)
    x = layers.RandomFlip("horizontal", seed=SEED, name="aug_flip")(inputs)
    x = layers.RandomRotation(0.08, seed=SEED, name="aug_rotate")(x)
    x = layers.RandomZoom(0.10, seed=SEED, name="aug_zoom")(x)
    x = layers.RandomBrightness(0.08, seed=SEED, name="aug_brightness")(x)

    # Preprocessing — DenseNet expects pixel scaling via its own preprocess_input
    x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="cast_float32")(x)
    x = layers.Lambda(
        keras.applications.densenet.preprocess_input, name="preprocess"
    )(x)

    # Backbone
    try:
        base = keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
        print(f"[{MODEL_NAME}] Loaded ImageNet pretrained weights.")
    except Exception as e:
        print(f"[{MODEL_NAME}] ImageNet weights failed, using random init. Reason:", e)
        base = keras.applications.DenseNet121(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling="avg",
        )

    base.trainable = False
    x = base(x, training=False)

    # Classification head
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(512, activation="relu", name="head_dense_1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="head_dense_2")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DenseNet121_classifier")
    return model, base


model, base_model = build_densenet121(NUM_CLASSES)
model.summary()

# ─────────────────────── Cell 7: Stage 1 — Head training ─────────────────────
# %%
def make_callbacks(stage: str):
    csv_path = OUTPUT_DIR / f"{MODEL_NAME}_{stage}_log.csv"
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
        ),
        callbacks.CSVLogger(str(csv_path)),
    ]


print(f"\n{'='*80}")
print(f"[{MODEL_NAME}] Stage 1: Training classification head (backbone frozen)")
print(f"{'='*80}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks("head"),
    verbose=1,
)

# ─────────────────── Cell 8: Stage 2 — Fine-tuning backbone ─────────────────
# %%
print(f"\n{'='*80}")
print(f"[{MODEL_NAME}] Stage 2: Fine-tuning upper backbone layers")
print(f"{'='*80}")

base_model.trainable = True
freeze_until = max(0, len(base_model.layers) - FINE_TUNE_LAST_N)

for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
for layer in base_model.layers[freeze_until:]:
    layer.trainable = True

print(f"  Total backbone layers : {len(base_model.layers)}")
print(f"  Frozen layers         : {freeze_until}")
print(f"  Trainable layers      : {len(base_model.layers) - freeze_until}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks("finetune"),
    verbose=1,
)

# ─────────────────── Cell 9: Training curves ─────────────────────────────────
# %%
def plot_training_curves(h_head, h_ft):
    history = {}
    for key in h_head.history:
        history[key] = list(h_head.history[key]) + list(h_ft.history.get(key, []))

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], label="train acc")
    ax1.plot(epochs, history["val_accuracy"], label="val acc")
    ax1.set_title(f"{MODEL_NAME} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["loss"], label="train loss")
    ax2.plot(epochs, history["val_loss"], label="val loss")
    ax2.set_title(f"{MODEL_NAME} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / f"{MODEL_NAME}_training_curves.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved training curves → {path}")


plot_training_curves(history_head, history_ft)

# ─────────────────── Cell 10: Evaluation ─────────────────────────────────────
# %%
def evaluate_model(model, test_ds, class_names):
    print(f"[{MODEL_NAME}] Evaluating on test set …")
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    print(f"[{MODEL_NAME}] Test accuracy : {acc:.4f}")
    print(f"[{MODEL_NAME}] Test macro-F1 : {mf1:.4f}")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", cbar=True)
    plt.title(f"{MODEL_NAME} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"{MODEL_NAME}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Saved confusion matrix → {cm_path}")

    return {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


metrics = evaluate_model(model, test_ds, class_names)

# ─────────────────── Cell 11: Export .h5 and .pkl ────────────────────────────
# %%
def export_artifacts(model, class_names, metrics):
    h5_path = OUTPUT_DIR / f"{MODEL_NAME}.h5"
    pkl_path = OUTPUT_DIR / f"{MODEL_NAME}.pkl"

    print(f"[{MODEL_NAME}] Saving H5 → {h5_path}")
    model.save(h5_path)

    bundle = {
        "model_name": MODEL_NAME,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "h5_path": str(h5_path),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        },
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pickle_mode": "metadata_only",
        "notes": "Load model from h5_path if pickled bytes unavailable.",
    }

    try:
        bundle["pickled_model_bytes"] = pickle.dumps(model)
        bundle["pickle_mode"] = "embedded_model_bytes"
        print(f"[{MODEL_NAME}] Embedded model bytes into PKL.")
    except Exception as e:
        bundle["pickled_model_bytes"] = None
        bundle["pickle_error"] = str(e)
        print(f"[{MODEL_NAME}] Full model pickling failed (metadata-only PKL).")
        print("Reason:", e)

    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[{MODEL_NAME}] Saved PKL → {pkl_path}")

    # artifact_index.json
    index = {}
    idx_path = OUTPUT_DIR / "artifact_index.json"
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)

    index[MODEL_NAME] = {
        "h5_path": str(h5_path),
        "pkl_path": str(pkl_path),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[{MODEL_NAME}] Updated artifact_index.json")

    return h5_path, pkl_path


h5_path, pkl_path = export_artifacts(model, class_names, metrics)

# ─────────────────── Cell 12: Inference helper ───────────────────────────────
# %%
SEVERITY_KEYWORDS = {
    "severe": ["cancer", "melanoma", "lupus", "vasculitis"],
    "moderate": [
        "eczema", "psoriasis", "rosacea", "keratosis",
        "bullous", "candidiasis", "lichen",
    ],
}


def infer_severity(disease_name: str, confidence: float):
    name = disease_name.lower()
    if any(k in name for k in SEVERITY_KEYWORDS["severe"]):
        base = "Severe"
    elif any(k in name for k in SEVERITY_KEYWORDS["moderate"]):
        base = "Moderate"
    else:
        base = "Mild"
    if confidence < 0.50:
        return "Mild"
    if confidence < 0.70 and base == "Severe":
        return "Moderate"
    return base


def predict_skin_disease(image_source, model=model, class_names=class_names):
    """
    Predict skin disease from a local file path or URL.

    Returns dict with keys: disease, confidence, severity.
    """
    source = str(image_source)
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        image = Image.open(source).convert("RGB")

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    disease = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    severity = infer_severity(disease, confidence)

    result = {
        "model_used": MODEL_NAME,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
    }
    print("Prediction:", result)
    return result


# Example: predict_skin_disease("/path/to/image.jpg")
# Example: predict_skin_disease("https://example.com/image.jpg")

print("\n✅ DenseNet121 training complete!")
print(f"   Accuracy : {metrics['accuracy']:.4f}")
print(f"   Macro-F1 : {metrics['macro_f1']:.4f}")
print(f"   H5  file : {h5_path}")
print(f"   PKL file : {pkl_path}")
