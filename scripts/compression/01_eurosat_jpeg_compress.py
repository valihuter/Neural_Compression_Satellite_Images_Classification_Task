#!/usr/bin/env python3
"""
JPEG Compression - EuroSAT Dataset

Dataset: eurosat RGB (64x64, 10 classes, 4,050 test images)
Codec: jpeg (PIL/Pillow)
Hardware: Local Mac CPU (~1ms/image)

Quality Levels:
    q1: quality=10  → ~0.29 BPP
    q2: quality=25  → ~0.55 BPP
    q3: quality=50  → ~1.00 BPP
    q4: quality=75  → ~1.31 BPP
    q5: quality=90  → ~1.70 BPP
    q6: quality=95  → ~2.00 BPP

Usage:
    python 01_eurosat_jpeg_compress.py --quality q1
    python 01_eurosat_jpeg_compress.py --quality all
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET = "eurosat"
IMAGE_SIZE = (64, 64)
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "eurosat" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "eurosat" / "comp" / "jpeg"

QUALITY_LEVELS = {
    "q1": 10,
    "q2": 25,
    "q3": 50,
    "q4": 75,
    "q5": 90,
    "q6": 95,
}

def compress_jpeg(input_path: Path, output_path: Path, quality: int):
    """Compress single image with JPEG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(input_path)
    img.save(output_path, "JPEG", quality=quality)

def main():
    parser = argparse.ArgumentParser(description="JPEG compression for EuroSAT")
    parser.add_argument("--quality", choices=["q1", "q2", "q3", "q4", "q5", "q6", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qualities = list(QUALITY_LEVELS.keys()) if args.quality == "all" else [args.quality]

    print(f"Dataset: {DATASET}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Qualities: {qualities}")

    if args.dry_run:
        print("[DRY RUN] No compression performed.")
        return

    for qname in qualities:
        quality = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"\n{qname}: quality={quality}")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.tif"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".jpg")
            compress_jpeg(img_path, out_path, quality)

        print(f"Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
