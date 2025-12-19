#!/usr/bin/env python3
"""
Cheng2020-attn Compression - EuroSAT Dataset

Dataset: EuroSAT RGB (64×64, 10 classes, 4,050 test images)
Codec: Cheng2020-attn (CompressAI)
Hardware: RunPod RTX 3090 GPU (~50ms/image)

Quality Levels (CompressAI quality index 1-6):
    q1: quality=1 → ~0.06 BPP (too aggressive for 64×64)
    q2: quality=2 → ~0.10 BPP (too aggressive for 64×64)
    q3: quality=3 → ~0.19 BPP
    q4: quality=4 → ~0.33 BPP
    q5: quality=5 → ~0.51 BPP
    q6: quality=6 → ~0.72 BPP

Note: q1-q2 produce severely degraded output for 64×64 images.
      Evaluation uses q3-q6 only.

Execution Environment:
    RunPod with NVIDIA RTX 3090 (24GB VRAM)
    pip install compressai torch torchvision

Usage:
    python eurosat_cheng2020.py --quality q3
    python eurosat_cheng2020.py --quality all
=============================================================================
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
OUTPUT_DIR = BASE_DIR / "data" / "eurosat" / "comp" / "cheng2020"

QUALITY_LEVELS = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "q5": 5,
    "q6": 6,
}

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
    parser = argparse.ArgumentParser(description="Cheng2020 compression for EuroSAT")
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
        print(f"{qname} (quality={quality})")

        model = cheng2020_attn(quality=quality, pretrained=True).eval().to(device)

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.tif"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".png")
            compress_cheng2020(img_path, out_path, model, device)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
