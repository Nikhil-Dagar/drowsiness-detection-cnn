"""
train_model.py
Trains a small CNN to classify a cropped eye image as Open or Closed.

Expected dataset folder structure (tf.keras.utils.image_dataset_from_directory
convention — download a public open/closed-eye dataset such as the MRL Eye
Dataset or Kaggle's "Driver Drowsiness Dataset" and arrange it like this):

    dataset/
        train/
            Open/
                img001.jpg ...
            Closed/
                img001.jpg ...
        val/
            Open/
                img001.jpg ...
            Closed/
                img001.jpg ...

Note: this uses tf.keras.utils.image_dataset_from_directory + Keras
augmentation layers instead of the old ImageDataGenerator, which was
removed in Keras 3 (bundled with TensorFlow 2.16+). Functionally
equivalent to the original ImageDataGenerator-based pipeline.

Usage:
    python src/train_model.py --data_dir dataset --epochs 20
"""

import argparse
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (24, 24)
BATCH_SIZE = 32


def build_datasets(data_dir):
    """Loads train/val datasets and applies rescaling + augmentation
    (replaces the old ImageDataGenerator pipeline)."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names  # e.g. ['Closed', 'Open']

    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(8 / 360),   # ~ rotation_range=8
        tf.keras.layers.RandomZoom(0.1),           # zoom_range=0.1
        tf.keras.layers.RandomTranslation(0.05, 0.05),  # width/height_shift_range=0.05
    ])

    train_ds = train_ds.map(
        lambda x, y: (rescale(augment(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model():
    """Small CNN — enough capacity for 24x24 grayscale eye crops."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[1], IMG_SIZE[0], 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),  # 0 = Closed, 1 = Open
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset",
                         help="Path to dataset root containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=str, default="models/eye_state_cnn.keras")
    args = parser.parse_args()

    train_ds, val_ds, class_names = build_datasets(args.data_dir)
    print("Class names (0/1 mapping, alphabetical):", class_names)

    model = build_model()
    model.summary()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(args.out, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest validation accuracy: {final_val_acc:.4f}")
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()
