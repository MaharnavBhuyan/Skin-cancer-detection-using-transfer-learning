"""
Case Study 5: Skin Cancer Detection using TensorFlow (Transfer Learning)
Classes: benign (0) vs malignant (1)

Dataset (Kaggle): fanconic/skin-cancer-malignant-vs-benign
Expected folder layout:
    DATA_DIR/
        train/benign, train/malignant
        test/benign,  test/malignant

Install:  pip install tensorflow pandas numpy matplotlib scikit-learn kagglehub
Run:      python skin_cancer_detection.py
"""

# ------------------------------------------------------------------
# 1. Importing Libraries
# ------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 10          # phase 1: train classifier head (base frozen)
EPOCHS_FINE = 10          # phase 2: fine-tune top layers of base
FINE_TUNE_LAYERS = 30     # number of top layers to unfreeze
AUTOTUNE = tf.data.AUTOTUNE

# Set DATA_DIR manually, or let the script download it via kagglehub.
DATA_DIR = os.environ.get("DATA_DIR", "")

if not DATA_DIR or not os.path.isdir(DATA_DIR):
    import kagglehub
    path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
    # locate the folder that contains train/ and test/
    for root, dirs, _ in os.walk(path):
        if "train" in dirs and "test" in dirs:
            DATA_DIR = root
            break

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
print("Using dataset at:", DATA_DIR)

# ------------------------------------------------------------------
# 3. Load Data
# ------------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    validation_split=0.2, subset="training", seed=SEED)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    validation_split=0.2, subset="validation", seed=SEED)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="binary",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_ds.class_names          # ['benign', 'malignant']
print("Classes:", class_names)

# ------------------------------------------------------------------
# 4. Data Exploration
# ------------------------------------------------------------------
counts = {}
for split, d in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    for c in class_names:
        counts[(split, c)] = len(os.listdir(os.path.join(d, c)))
df_counts = pd.Series(counts).unstack()
print("\nImage counts:\n", df_counts)

df_counts.plot(kind="bar", figsize=(6, 4), title="Class distribution")
plt.ylabel("Images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=120)
plt.show()

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i].numpy()[0])])
        plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.show()

# Class weights (handles imbalance)
train_labels = np.concatenate([y.numpy().ravel() for _, y in train_ds])
weights = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=train_labels)
class_weight = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weight)

# Performance pipeline
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ------------------------------------------------------------------
# 5. Model: EfficientNetB0 Transfer Learning
# ------------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="augmentation")

base_model = EfficientNetB0(include_top=False, weights="imagenet",
                            input_shape=IMG_SIZE + (3,))
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=False)          # EfficientNet has built-in preprocessing (expects 0-255)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs, name="skin_cancer_effnetb0")
model.summary()

metrics = [
    "accuracy",
    keras.metrics.AUC(name="auc"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                      min_lr=1e-7, verbose=1),
    ModelCheckpoint("best_skin_cancer_model.keras", monitor="val_auc",
                    mode="max", save_best_only=True, verbose=1),
]

# ------------------------------------------------------------------
# 6. Phase 1: Train the classifier head
# ------------------------------------------------------------------
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=metrics)

history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                     class_weight=class_weight, callbacks=callbacks)

# ------------------------------------------------------------------
# 7. Phase 2: Fine-tune top layers of the base model
# ------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers:                 # keep BatchNorm frozen
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=metrics)

history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=EPOCHS_HEAD + EPOCHS_FINE,
                     initial_epoch=history1.epoch[-1] + 1,
                     class_weight=class_weight, callbacks=callbacks)

# ------------------------------------------------------------------
# 8. Training Curves
# ------------------------------------------------------------------
def merge(h1, h2, key):
    return h1.history[key] + h2.history[key]

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for a, key, title in zip(ax, ["accuracy", "loss", "auc"],
                         ["Accuracy", "Loss", "AUC"]):
    a.plot(merge(history1, history2, key), label="train")
    a.plot(merge(history1, history2, "val_" + key), label="val")
    a.axvline(len(history1.history[key]) - 1, color="gray", ls="--", label="fine-tune start")
    a.set_title(title)
    a.set_xlabel("Epoch")
    a.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
plt.show()

# ------------------------------------------------------------------
# 9. Evaluation on Test Set
# ------------------------------------------------------------------
model = keras.models.load_model("best_skin_cancer_model.keras")
results = model.evaluate(test_ds, verbose=0)
print("\nTest results:")
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:\n",
      classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120)
plt.show()

# ------------------------------------------------------------------
# 10. Save Final Model
# ------------------------------------------------------------------
model.save("skin_cancer_final_model.keras")
print("Model saved: skin_cancer_final_model.keras")

# ------------------------------------------------------------------
# 11. Predict on a Single Image
# ------------------------------------------------------------------
def predict_image(img_path, model=model, threshold=0.5):
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(keras.utils.img_to_array(img), axis=0)   # 0-255 range
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = class_names[int(prob >= threshold)]
    conf = prob if prob >= threshold else 1 - prob
    plt.imshow(img)
    plt.title(f"{label} ({conf:.2%})")
    plt.axis("off")
    plt.show()
    return label, conf

# Example:
# predict_image("path/to/image.jpg")

# Visualise some test predictions
plt.figure(figsize=(12, 8))
for images, labels in test_ds.take(1):
    probs = model.predict(images, verbose=0).ravel()
    for i in range(min(12, len(images))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred = class_names[int(probs[i] >= 0.5)]
        true = class_names[int(labels[i].numpy()[0])]
        plt.title(f"T: {true} | P: {pred}", color="green" if pred == true else "red")
        plt.axis("off")
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=120)
plt.show()

print("\nDisclaimer: educational project only, not a medical diagnostic tool.")
