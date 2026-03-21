import tensorflow as tf
import tf_keras as keras

IMAGE_SIZE = 224
NUM_CLASSES = 22

inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
x = keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(inputs)
x = keras.layers.Lambda(keras.applications.mobilenet_v2.preprocess_input)(x)

base_model = keras.applications.MobileNetV2(
    include_top=False, weights=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling='avg'
)
base_model._name = "functional"
x = base_model(x, training=False)

x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dropout(0.35)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.175)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print("Attempting strict load (no skip_mismatch)...")
try:
    model.load_weights("model.h5")
    print("SUCCESS: Loaded perfectly without by_name or skip_mismatch!")
except Exception as e:
    print(f"STRICT LOAD ERROR: {e}")

print("\nAttempting by_name load (no skip_mismatch)...")
try:
    model.load_weights("model.h5", by_name=True)
    print("SUCCESS: Loaded perfectly with by_name=True")
except Exception as e:
    print(f"BY_NAME LOAD ERROR: {e}")
