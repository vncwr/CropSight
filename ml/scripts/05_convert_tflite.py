"""
05_convert_tflite.py — Convert trained model to TFLite

Converts the Keras model to TFLite with dynamic range quantization.
Sanity-checks the quantized model against the float model on test images.
Generates labels.txt and copies both to the Flutter assets directory.
"""

import os
import argparse
import shutil
import numpy as np
import tensorflow as tf

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ML_DIR, "data", "splits")
MODEL_DIR = os.path.join(ML_DIR, "models")
ASSETS_DIR = os.path.join(os.path.dirname(ML_DIR), "assets", "model")
IMG_SIZE = (224, 224)


def convert(model_dir, data_dir, assets_dir):
    print(f"{'='*60}")
    print(f"  CropSight — TFLite Conversion")
    print(f"{'='*60}")

    model_path = os.path.join(model_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model not found at {model_path}")
        return

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Convert with dynamic range quantization
    print("Converting to TFLite with dynamic range quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Keep float32 input/output for easier integration
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = os.path.join(model_dir, 'rice_disease_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"  TFLite model saved: {tflite_path}")
    print(f"  Model size: {model_size_mb:.2f} MB")

    # Load class names
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: read from test directory
        test_dir = os.path.join(data_dir, 'test')
        class_names = sorted(os.listdir(test_dir))

    # Save labels.txt
    labels_path = os.path.join(model_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  Labels saved: {labels_path}")
    print(f"  Classes: {', '.join(class_names)}")

    # Sanity check: compare float vs quantized predictions
    print(f"\n{'─'*60}")
    print(f"  Sanity Check: Float vs Quantized Predictions")
    print(f"{'─'*60}")

    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=1,
            shuffle=True,
            seed=42,
        )

        normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)

        # Load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        matches = 0
        total = 0
        max_checks = 50

        for images, _ in test_ds.take(max_checks):
            img_normalized = normalization(images).numpy()

            # Float model prediction
            float_pred = model.predict(img_normalized, verbose=0)
            float_class = np.argmax(float_pred[0])

            # TFLite prediction
            interpreter.set_tensor(input_details[0]['index'], img_normalized.astype(np.float32))
            interpreter.invoke()
            tflite_pred = interpreter.get_tensor(output_details[0]['index'])
            tflite_class = np.argmax(tflite_pred[0])

            if float_class == tflite_class:
                matches += 1
            total += 1

        agreement = matches / total * 100
        print(f"\n  Checked {total} images")
        print(f"  Agreement: {matches}/{total} ({agreement:.1f}%)")

        if agreement < 95:
            print(f"  ⚠️  WARNING: Agreement below 95%! Quantization may have degraded accuracy.")
        else:
            print(f"  ✓  Good — quantized model agrees with float model.")

    # Copy to Flutter assets
    print(f"\n{'─'*60}")
    print(f"  Copying to Flutter assets")
    print(f"{'─'*60}")

    os.makedirs(assets_dir, exist_ok=True)

    asset_tflite = os.path.join(assets_dir, 'model.tflite')
    asset_labels = os.path.join(assets_dir, 'labels.txt')

    shutil.copy2(tflite_path, asset_tflite)
    shutil.copy2(labels_path, asset_labels)

    print(f"  {asset_tflite}")
    print(f"  {asset_labels}")
    print(f"\n{'='*60}")
    print(f"  Done! Model is ready for the Flutter app.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to TFLite")
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--data_dir", default=DATA_ROOT)
    parser.add_argument("--assets_dir", default=ASSETS_DIR)
    args = parser.parse_args()
    convert(args.model_dir, args.data_dir, args.assets_dir)
