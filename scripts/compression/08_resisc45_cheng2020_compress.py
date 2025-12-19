#!/usr/bin/env python3
"""
Cheng2020-attn Compression - RESISC45 Subset Dataset

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Codec: Cheng2020-attn (CompressAI)
Hardware: RunPod RTX 3090 GPU (~50ms/image)

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Quality Levels (CompressAI quality index 1-6):
    q1: quality=1 → N/A (not used for evaluation)
    q2: quality=2 → N/A (not used for evaluation)
    q3: quality=3 → ~0.30 BPP
    q4: quality=4 → ~0.49 BPP
    q5: quality=5 → ~0.71 BPP
    q6: quality=6 → ~0.98 BPP

Note: q1-q2 produce very low BPP but are included in compression.
      Evaluation focuses on q3-q6.

Execution Environment:
    RunPod with NVIDIA RTX 3090 (24GB VRAM)
    pip install compressai torch torchvision

Usage:
    python resisc45_cheng2020.py --quality q3
    python resisc45_cheng2020.py --quality all
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
OUTPUT_DIR = BASE_DIR / "data" / "resisc45" / "comp" / "cheng2020"

QUALITY_LEVELS = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "q5": 5,
    "q6": 6,
}

# Measured BPP values (December 2025)
MEASURED_BPP = {"q1": None, "q2": None, "q3": 0.304, "q4": 0.494, "q5": 0.709, "q6": 0.982}

def compress_cheng2020(input_path: Path, output_path: Path, model, device):
    """Compress single image with Cheng2020-attn."""
    import torch
    from torchvision import transforms

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.compress(x)
        rec = model.decompress(out["strings"], out["shape"])

    rec_img = transforms.ToPILImage()(rec["x_hat"].squeeze().clamp(0, 1).cpu())
    rec_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Cheng2020 compression for RESISC45")
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
            bpp = MEASURED_BPP[q]
            bpp_str = f"{bpp:.3f}" if bpp else "N/A"
            print(f"  {q}: quality={QUALITY_LEVELS[q]} → {bpp_str} BPP")
        return

    try:
        import torch
        from compressai.zoo import cheng2020_attn
    except ImportError:
        print("ERROR: CompressAI not installed.")
        print("Install with: pip install compressai")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for qname in qualities:
        quality = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        bpp = MEASURED_BPP[qname]
        bpp_str = f"{bpp:.3f}" if bpp else "N/A"
        print(f"
{qname} (quality={quality}, measured BPP={bpp_str})")

        model = cheng2020_attn(quality=quality, pretrained=True).eval().to(device)

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".png")
            compress_cheng2020(img_path, out_path, model, device)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
