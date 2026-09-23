"""
04_evaluate.py — Evaluate the trained model

Reports accuracy, per-class precision/recall/F1, and confusion matrix
on the test split. Flags any class with F1 < 0.7.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ML_DIR, "data", "splits")
MODEL_DIR = os.path.join(ML_DIR, "models")
IMG_SIZE = (224, 224)


def evaluate(data_dir, model_dir):
    print(f"{'='*60}")
    print(f"  CropSight — Model Evaluation")
    print(f"{'='*60}")

    test_dir = os.path.join(data_dir, 'test')
    model_path = os.path.join(model_dir, 'best_model.keras')

    if not os.path.exists(model_path):
        print(f"\n  ERROR: Model not found at {model_path}")
        print("  Run 03_train.py first.")
        return

    # Load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=32,
        label_mode='categorical',
        shuffle=False,
    )

    class_names = test_ds.class_names
    print(f"\nClasses: {', '.join(class_names)}")
    print(f"Test samples: {sum(1 for _ in test_ds.unbatch())}")

    # Normalize
    normalization = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    test_ds_norm = test_ds.map(lambda x, y: (normalization(x), y))

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Predict
    print("Running predictions...")
    predictions = model.predict(test_ds_norm, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Get true labels
    y_true = np.concatenate([
        np.argmax(y.numpy(), axis=1) for _, y in test_ds
    ])

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4
    )

    print(f"\n{'─'*60}")
    print(f"  Results")
    print(f"{'─'*60}")
    print(f"\n  Overall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.1f}%)")

    print(f"\n  {'Class':25s} | {'Precision':>10s} | {'Recall':>8s} | {'F1':>8s} | {'Support':>8s}")
    print(f"  {'─'*25}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}")
    flagged = []
    for cls in class_names:
        m = report[cls]
        flag = " ⚠️" if m['f1-score'] < 0.7 else ""
        print(f"  {cls:25s} | {m['precision']:10.4f} | {m['recall']:8.4f} | {m['f1-score']:8.4f} | {int(m['support']):8d}{flag}")
        if m['f1-score'] < 0.7:
            flagged.append(cls)

    if flagged:
        print(f"\n  ⚠️  WARNING: Low F1 classes: {', '.join(flagged)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':25s}   {'  '.join(f'{c[:8]:>8s}' for c in class_names)}")
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:25s}   {'  '.join(f'{v:8d}' for v in row)}")

    # Save report
    report_path = os.path.join(model_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write("# CropSight — Model Evaluation Report\n\n")
        f.write(f"## Overall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.1f}%)\n\n")
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|----------|\n")
        for cls in class_names:
            m = report[cls]
            flag = " ⚠️" if m['f1-score'] < 0.7 else ""
            f.write(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f}{flag} | {int(m['support'])} |\n")
        f.write(f"\n## Confusion Matrix\n\n```\n")
        f.write(f"{'':25s}  {'  '.join(f'{c[:10]:>10s}' for c in class_names)}\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]:25s}  {'  '.join(f'{v:10d}' for v in row)}\n")
        f.write("```\n")

    print(f"\n  Report saved to: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--data_dir", default=DATA_ROOT)
    parser.add_argument("--model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_dir)
