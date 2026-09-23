"""
03_train.py — Train MobileNetV2 on Rice Leaf Diseases

Fine-tunes MobileNetV2 (pretrained on ImageNet) for 3-class rice disease
classification: Bacterialblight, Brownspot, Leafsmut.

Strategy:
  1. Freeze base → train classification head (Adam LR=1e-3, up to 20 epochs)
  2. Unfreeze top 30 layers → fine-tune (Adam LR=1e-5, up to 15 epochs)

Input: 224x224x3, normalized to [-1, 1] (MobileNetV2 convention)
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ML_DIR, "data", "splits")
MODEL_DIR = os.path.join(ML_DIR, "models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def create_model(num_classes):
    """Build MobileNetV2 with custom classification head."""
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def train(data_dir, model_dir):
    print(f"{'='*60}")
    print(f"  CropSight — Model Training")
    print(f"{'='*60}")

    os.makedirs(model_dir, exist_ok=True)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"\n  ERROR: Train directory not found: {train_dir}")
        print("  Run 02_prepare_splits.py first.")
        return

    # Load datasets — images are resized to 224x224 from 300x300
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {', '.join(class_names)}")

    # Normalize pixels to [-1, 1] for MobileNetV2
    normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # Prefetch for performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Build model
    model, base_model = create_model(num_classes)

    model_path = os.path.join(model_dir, 'best_model.keras')

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # Phase 1: Train classification head only
    print(f"\n{'─'*60}")
    print(f"  Phase 1: Training classification head (base frozen)")
    print(f"{'─'*60}")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune top layers
    print(f"\n{'─'*60}")
    print(f"  Phase 2: Fine-tuning top 30 layers")
    print(f"{'─'*60}")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
    )

    # Save class names for later use
    labels_path = os.path.join(model_dir, 'class_names.txt')
    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print(f"\n  Model saved to: {model_path}")
    print(f"  Class names saved to: {labels_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rice disease classifier")
    parser.add_argument("--data_dir", default=DATA_ROOT)
    parser.add_argument("--model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)
