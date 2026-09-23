"""
01_explore_dataset.py — Explore the Rice Leaf Diseases Dataset

Scans the dataset directory, counts images per class, checks dimensions
and formats, flags corrupt files, and generates a data card.

Dataset: Mendeley Rice Leaf Diseases (3 classes)
  - Bacterialblight (1,604 images)
  - Brownspot (1,620 images) 
  - Leafsmut (1,460 images)
"""

import os
import argparse
from collections import defaultdict
from PIL import Image

# Default path to the extracted dataset
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "mendeley_rice", "extracted", "rice leaf diseases dataset"
)


def explore_dataset(data_dir):
    print(f"{'='*60}")
    print(f"  CropSight — Dataset Explorer")
    print(f"{'='*60}")
    print(f"\nDataset path: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"\n  ERROR: Directory does not exist: {data_dir}")
        print("  Make sure the dataset is extracted correctly.")
        return

    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not classes:
        print("\n  ERROR: No class subdirectories found.")
        return

    print(f"\nFound {len(classes)} classes: {', '.join(classes)}\n")

    class_counts = {}
    class_dims = defaultdict(set)
    corrupt_files = []
    total = 0

    for c in classes:
        class_dir = os.path.join(data_dir, c)
        count = 0
        extensions = defaultdict(int)

        for img_name in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            ext = os.path.splitext(img_name)[1].lower()
            extensions[ext] += 1

            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Re-open to get actual dimensions (verify closes the file)
                with Image.open(img_path) as img:
                    class_dims[c].add(img.size)
                count += 1
            except Exception as e:
                corrupt_files.append((img_path, str(e)))
                print(f"  CORRUPT: {img_path} — {e}")

        class_counts[c] = count
        total += count

        # Summarize dimensions for this class
        dims = class_dims[c]
        if len(dims) == 1:
            dim_str = f"{list(dims)[0][0]}x{list(dims)[0][1]}"
        else:
            dim_str = f"{len(dims)} unique sizes"

        ext_str = ", ".join(f"{k}: {v}" for k, v in sorted(extensions.items()))
        print(f"  {c:25s} | {count:5d} images | {dim_str:20s} | {ext_str}")

    print(f"\n  {'TOTAL':25s} | {total:5d} images")

    if corrupt_files:
        print(f"\n  WARNING: {len(corrupt_files)} corrupt files found!")
        for path, err in corrupt_files[:10]:
            print(f"    - {path}: {err}")
        if len(corrupt_files) > 10:
            print(f"    ... and {len(corrupt_files) - 10} more")
    else:
        print(f"\n  No corrupt files found.")

    # Generate data card
    ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    card_path = os.path.join(ml_dir, "data_card.md")

    with open(card_path, "w") as f:
        f.write("# CropSight — Data Card\n\n")
        f.write("## Source\n")
        f.write("- **Dataset**: Rice Leaf Diseases Dataset\n")
        f.write("- **Source**: Mendeley Data\n")
        f.write("- **License**: CC BY 4.0\n")
        f.write("- **URL**: https://data.mendeley.com/datasets/vwv3nry3wr/1\n\n")
        f.write("## Class Distribution\n\n")
        f.write("| Class | Images | Dimensions |\n")
        f.write("|-------|--------|------------|\n")
        for c in classes:
            dims = class_dims[c]
            if len(dims) == 1:
                dim_str = f"{list(dims)[0][0]}×{list(dims)[0][1]}"
            else:
                dim_str = f"{len(dims)} unique sizes"
            f.write(f"| {c} | {class_counts[c]} | {dim_str} |\n")
        f.write(f"| **Total** | **{total}** | |\n\n")
        f.write(f"## Quality\n")
        f.write(f"- Corrupt files: {len(corrupt_files)}\n")
        f.write(f"- Format: JPEG\n")

    print(f"\n  Data card saved to: {card_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore the rice disease dataset")
    parser.add_argument("--data_dir", default=DATA_ROOT, help="Path to dataset root")
    args = parser.parse_args()
    explore_dataset(args.data_dir)
