"""
02_prepare_splits.py — Split the Rice Leaf Diseases Dataset

Splits the dataset into train/val/test (80/10/10) with stratification.
Applies augmentation to training set only.

Dataset: 4,684 images across 3 classes (300x300 JPEG)
  - Bacterialblight: 1,604
  - Brownspot: 1,620
  - Leafsmut: 1,460

IMPORTANT: This dataset appears to already be all original images (no
pre-augmented copies with "aug" in filename). We split everything, then
augment the training split ourselves.
"""

import os
import argparse
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict

# Paths
ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ML_DIR, "data", "raw", "mendeley_rice", "extracted", "rice leaf diseases dataset")
OUTPUT_DIR = os.path.join(ML_DIR, "data", "splits")

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def augment_image(img):
    """Generate augmented versions of an image for training."""
    # Ensure RGB mode (some JPEGs have alpha channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    augmented = []

    # Rotation ±30°
    augmented.append(img.rotate(30, fillcolor=(0, 0, 0)))
    augmented.append(img.rotate(-30, fillcolor=(0, 0, 0)))

    # Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Brightness variations
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(1.2))
    augmented.append(enhancer.enhance(0.8))

    # Contrast variations
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(1.2))
    augmented.append(enhancer.enhance(0.8))

    # Slight Gaussian blur
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=1)))

    return augmented


def prepare_splits(data_dir, output_dir):
    print(f"{'='*60}")
    print(f"  CropSight — Dataset Splitter")
    print(f"{'='*60}")
    print(f"\nSource: {data_dir}")
    print(f"Output: {output_dir}")

    if not os.path.exists(data_dir):
        print(f"\n  ERROR: Source directory not found: {data_dir}")
        return

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print(f"\nClasses: {', '.join(classes)}")

    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create split directories
    for split in ['train', 'val', 'test']:
        for c in classes:
            os.makedirs(os.path.join(output_dir, split, c), exist_ok=True)

    random.seed(SEED)
    stats = defaultdict(lambda: defaultdict(int))

    for c in classes:
        class_dir = os.path.join(data_dir, c)
        images = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Shuffle deterministically
        random.shuffle(images)

        n = len(images)
        n_test = max(1, int(n * TEST_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_train = n - n_test - n_val

        test_imgs = images[:n_test]
        val_imgs = images[n_test:n_test + n_val]
        train_imgs = images[n_test + n_val:]

        # Copy test (no augmentation)
        for img_name in test_imgs:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(output_dir, 'test', c, img_name)
            shutil.copy2(src, dst)
            stats['test'][c] += 1

        # Copy val (no augmentation)
        for img_name in val_imgs:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(output_dir, 'val', c, img_name)
            shutil.copy2(src, dst)
            stats['val'][c] += 1

        # Copy train + augment
        for img_name in train_imgs:
            src = os.path.join(class_dir, img_name)
            dst = os.path.join(output_dir, 'train', c, img_name)
            shutil.copy2(src, dst)
            stats['train'][c] += 1

            # Augment
            try:
                with Image.open(src) as img:
                    for j, aug_img in enumerate(augment_image(img)):
                        aug_name = f"aug{j}_{img_name}"
                        aug_path = os.path.join(output_dir, 'train', c, aug_name)
                        aug_img.save(aug_path, 'JPEG', quality=95)
                        stats['train_aug'][c] += 1
            except Exception as e:
                print(f"  Warning: Failed to augment {img_name}: {e}")

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  Split Summary")
    print(f"{'─'*60}")
    print(f"  {'Class':25s} | {'Train':>7s} | {'+ Aug':>7s} | {'Val':>5s} | {'Test':>5s}")
    print(f"  {'─'*25}-+-{'─'*7}-+-{'─'*7}-+-{'─'*5}-+-{'─'*5}")
    for c in classes:
        train_n = stats['train'][c]
        aug_n = stats['train_aug'][c]
        val_n = stats['val'][c]
        test_n = stats['test'][c]
        print(f"  {c:25s} | {train_n:7d} | {aug_n:7d} | {val_n:5d} | {test_n:5d}")

    total_train = sum(stats['train'].values())
    total_aug = sum(stats['train_aug'].values())
    total_val = sum(stats['val'].values())
    total_test = sum(stats['test'].values())
    print(f"  {'─'*25}-+-{'─'*7}-+-{'─'*7}-+-{'─'*5}-+-{'─'*5}")
    print(f"  {'TOTAL':25s} | {total_train:7d} | {total_aug:7d} | {total_val:5d} | {total_test:5d}")
    print(f"\n  Total training images (orig + aug): {total_train + total_aug}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--data_dir", default=DATA_ROOT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    prepare_splits(args.data_dir, args.output_dir)
