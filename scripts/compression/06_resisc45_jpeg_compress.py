#!/usr/bin/env python3
"""
JPEG Compression - RESISC45 Subset Dataset

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Codec: JPEG (PIL/Pillow)
Hardware: Local Mac CPU (~1ms/image)

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Quality Levels:
    q1: quality=10  → ~0.30 BPP
    q2: quality=25  → ~0.66 BPP
    q3: quality=50  → ~1.20 BPP
    q4: quality=75  → ~1.55 BPP
    q5: quality=90  → ~1.90 BPP
    q6: quality=95  → ~2.23 BPP

Usage:
    python resisc45_jpeg.py --quality q1
    python resisc45_jpeg.py --quality all
=============================================================================
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET = "resisc45"
IMAGE_SIZE = (256, 256)
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "resisc45" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "resisc45" / "comp" / "jpeg"

QUALITY_LEVELS = {
    "q1": 10,
    "q2": 25,
    "q3": 50,
    "q4": 75,
    "q5": 90,
    "q6": 95,
}

# Measured BPP values (December 2025)
MEASURED_BPP = {"q1": 0.302, "q2": 0.658, "q3": 1.201, "q4": 1.549, "q5": 1.900, "q6": 2.230}

def compress_jpeg(input_path: Path, output_path: Path, quality: int):
    """Compress single image with JPEG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(input_path)
    img.save(output_path, "JPEG", quality=quality)

def main():
    parser = argparse.ArgumentParser(description="JPEG compression for RESISC45")
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
            print(f"  {q}: quality={QUALITY_LEVELS[q]} → {MEASURED_BPP[q]:.3f} BPP")
        return

    for qname in qualities:
        quality = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"
{qname} (quality={quality}, measured BPP={MEASURED_BPP[qname]:.3f})")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".jpg")
            compress_jpeg(img_path, out_path, quality)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
