#!/usr/bin/env python3
"""
JPEG2000 Compression - RESISC45 Subset Dataset

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Codec: JPEG2000 (OpenCV with OpenJPEG backend)
Hardware: Local Mac CPU (~10ms/image)

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Quality Levels (compression ratio × 1000):
    q1: 10  → ~0.24 BPP
    q2: 15  → ~0.35 BPP
    q3: 25  → ~0.59 BPP
    q4: 45  → ~0.95 BPP
    q5: 70  → ~1.70 BPP
    q6: 97  → ~2.39 BPP

Usage:
    python resisc45_jpeg2000.py --quality q1
    python resisc45_jpeg2000.py --quality all
=============================================================================
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

# Configuration
DATASET = "resisc45"
IMAGE_SIZE = (256, 256)
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "resisc45" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "resisc45" / "comp" / "jpeg2000"

QUALITY_LEVELS = {
    "q1": 10,
    "q2": 15,
    "q3": 25,
    "q4": 45,
    "q5": 70,
    "q6": 97,
}

# Measured BPP values (December 2025)
MEASURED_BPP = {"q1": 0.235, "q2": 0.351, "q3": 0.592, "q4": 0.950, "q5": 1.702, "q6": 2.386}

def compress_jpeg2000(input_path: Path, output_path: Path, quality: int):
    """Compress single image with JPEG2000."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(input_path))
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality])

def main():
    parser = argparse.ArgumentParser(description="JPEG2000 compression for RESISC45")
    parser.add_argument("--quality", choices=["q1", "q2", "q3", "q4", "q5", "q6", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qualities = list(QUALITY_LEVELS.keys()) if args.quality == "all" else [args.quality]

    print(f"Dataset: {DATASET}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Qualities: {qualities}")

    if args.dry_run:
        print("\n[DRY RUN] Measured BPP values:")
        for q in qualities:
            print(f"  {q}: compression_x1000={QUALITY_LEVELS[q]} → {MEASURED_BPP[q]:.3f} BPP")
        return

    for qname in qualities:
        quality = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"
{qname} (compression_x1000={quality}, measured BPP={MEASURED_BPP[qname]:.3f})")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".jp2")
            compress_jpeg2000(img_path, out_path, quality)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
