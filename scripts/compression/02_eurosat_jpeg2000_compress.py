#!/usr/bin/env python3
"""
JPEG2000 Compression - EuroSAT Dataset

Dataset: EuroSAT RGB (64×64, 10 classes, 4,050 test images)
Codec: JPEG2000 (OpenCV with OpenJPEG backend)
Hardware: Local Mac CPU (~10ms/image)

Quality Levels (compression ratio × 1000):
    q1: 10  → ~0.19 BPP
    q2: 15  → ~0.28 BPP
    q3: 25  → ~0.47 BPP
    q4: 45  → ~0.91 BPP
    q5: 70  → ~1.48 BPP
    q6: 97  → ~2.05 BPP

Usage:
    python eurosat_jpeg2000.py --quality q1
    python eurosat_jpeg2000.py --quality all
=============================================================================
"""

import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

# Configuration
DATASET = "eurosat"
IMAGE_SIZE = (64, 64)
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "eurosat" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "eurosat" / "comp" / "jpeg2000"

QUALITY_LEVELS = {
    "q1": 10,
    "q2": 15,
    "q3": 25,
    "q4": 45,
    "q5": 70,
    "q6": 97,
}

def compress_jpeg2000(input_path: Path, output_path: Path, quality: int):
    """Compress single image with JPEG2000."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(input_path))
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality])

def main():
    parser = argparse.ArgumentParser(description="JPEG2000 compression for EuroSAT")
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
        print(f"{qname} (compression_x1000={quality})")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.tif"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".jp2")
            compress_jpeg2000(img_path, out_path, quality)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()