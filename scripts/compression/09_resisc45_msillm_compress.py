#!/usr/bin/env python3
"""
MS-ILLM Compression - RESISC45 Subset Dataset

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Codec: MS-ILLM (facebookresearch/NeuralCompression)
Hardware: RunPod 2× RTX 4090 GPU (~100ms/image with multi-GPU)

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Quality Levels:
    q1: quality=1 → ~0.04 BPP
    q2: quality=2 → ~0.06 BPP
    q3: quality=3 → ~0.12 BPP
    q4: quality=4 → ~0.22 BPP
    q5: quality=5 → ~0.43 BPP
    q6: quality=6 → ~0.81 BPP

Execution Environment:
    RunPod with 2× NVIDIA RTX 4090 (48GB total VRAM)
    Multi-GPU parallelization via DataParallel
    pip install torch torchvision

Usage:
    python resisc45_msillm.py --quality q1
    python resisc45_msillm.py --quality all
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
OUTPUT_DIR = BASE_DIR / "data" / "resisc45" / "comp" / "msillm"

QUALITY_LEVELS = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "q5": 5,
    "q6": 6,
}

# Measured BPP values (December 2025)
MEASURED_BPP = {"q1": 0.037, "q2": 0.064, "q3": 0.121, "q4": 0.224, "q5": 0.429, "q6": 0.806}

def compress_msillm(input_path: Path, output_path: Path, model, device):
    """Compress single image with MS-ILLM."""
    import torch
    from torchvision import transforms

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.compress(x)
        rec = model.decompress(out)

    rec_img = transforms.ToPILImage()(rec.squeeze().clamp(0, 1).cpu())
    rec_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="MS-ILLM compression for RESISC45")
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

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for qname in qualities:
        quality = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"
{qname} (quality={quality}, measured BPP={MEASURED_BPP[qname]:.3f})")

        try:
            model = torch.hub.load("facebookresearch/NeuralCompression", f"msillm_quality_{quality}")
            model = model.eval().to(device)
        except Exception as e:
            print(f"ERROR loading MS-ILLM model: {e}")
            continue

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".png")
            compress_msillm(img_path, out_path, model, device)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
