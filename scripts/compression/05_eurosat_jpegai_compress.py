#!/usr/bin/env python3
"""
JPEG-AI Compression - EuroSAT Dataset

Dataset: EuroSAT RGB (64×64, 10 classes, 4,050 test images)
Codec: JPEG-AI (ISO/IEC 6048-1:2025, Reference Software v6.0)
Hardware: RunPod RTX 3090 GPU, sequential processing (~200ms/image)

Quality Levels (target BPP × 100):
    q1: 13  → 0.13 BPP
    q2: 25  → 0.25 BPP
    q3: 35  → 0.35 BPP (actual ~0.35)
    q4: 45  → 0.45 BPP
    q5: 55  → 0.55 BPP
    q6: 72  → 0.72 BPP

IMPORTANT - 64×64 Workaround:
    JPEG-AI requires minimum 128×128 (actually 160×160 with eICCI filter).
    Workaround: Upscale 64×64 → 192×192 → compress → decode → downscale 64×64
    This introduces interpolation artifacts → 88% accuracy ceiling!
    Config: cfg/tools_off.json (disables unsupported features for small images)

Execution Environment:
    RunPod with NVIDIA RTX 3090 (24GB VRAM)
    JPEG-AI Reference Software from official GitLab
    Sequential processing (not parallelizable due to GPU memory)

Usage:
    python eurosat_jpegai.py --quality q1
    python eurosat_jpegai.py --quality all

Note: This is a DOCUMENTATION script. Actual compression was performed
on RunPod using cloud-gpu/compress_eurosat.py with GPU acceleration.
=============================================================================
"""

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
DATASET = "eurosat"
IMAGE_SIZE = (64, 64)
UPSCALE_SIZE = (192, 192)  # 3x upscale to meet JPEG-AI minimum

BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "eurosat" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "eurosat" / "comp" / "jpegai"

# JPEG-AI paths (RunPod)
JPEGAI_DIR = Path("/workspace/jpeg-ai-reference-software")

QUALITY_LEVELS = {
    "q1": 13,   # 0.13 BPP
    "q2": 25,   # 0.25 BPP
    "q3": 35,   # 0.35 BPP
    "q4": 45,   # 0.45 BPP
    "q5": 55,   # 0.55 BPP
    "q6": 72,   # 0.72 BPP
}

def compress_jpegai(input_path: Path, output_path: Path, target_bpp_x100: int, jpegai_dir: Path):
    """
    Compress single image with JPEG-AI.
    
    Workflow for 64×64 images:
    1. Upscale to 192×192 (NEAREST to preserve pixels)
    2. Encode with JPEG-AI
    3. Decode
    4. Downscale back to 64×64 (BICUBIC)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.mkdtemp())
    upscaled_input = temp_dir / "input_192.png"
    bitstream = temp_dir / "compressed.bin"
    upscaled_output = temp_dir / "output_192.png"
    
    try:
        # Step 1: Upscale
        img = Image.open(input_path)
        upscaled = img.resize(UPSCALE_SIZE, Image.NEAREST)
        upscaled.save(upscaled_input)
        
        # Step 2: Encode (GPU mode)
        encode_cmd = [
            'python', '-m', 'src.reco.coders.encoder',
            str(upscaled_input), str(bitstream),
            '--set_target_bpp', str(target_bpp_x100),
            '-target_device', 'gpu',
            '--cfg', 'cfg/tools_off.json', 'cfg/profiles/base.json'
        ]
        result = subprocess.run(encode_cmd, capture_output=True, cwd=str(jpegai_dir), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Encode failed: {result.stderr.decode()[:200]}")
        
        # Step 3: Decode (GPU mode)
        decode_cmd = [
            'python', '-m', 'src.reco.coders.decoder',
            str(bitstream), str(upscaled_output),
            '-target_device', 'gpu'
        ]
        result = subprocess.run(decode_cmd, capture_output=True, cwd=str(jpegai_dir), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Decode failed: {result.stderr.decode()[:200]}")
        
        # Step 4: Downscale
        decoded = Image.open(upscaled_output)
        downscaled = decoded.resize(IMAGE_SIZE, Image.BICUBIC)
        downscaled.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="JPEG-AI compression for EuroSAT")
    parser.add_argument("--quality", choices=["q1", "q2", "q3", "q4", "q5", "q6", "all"], required=True)
    parser.add_argument("--jpegai-dir", type=str, default=str(JPEGAI_DIR))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qualities = list(QUALITY_LEVELS.keys()) if args.quality == "all" else [args.quality]
    jpegai_dir = Path(args.jpegai_dir)

    print(f"Dataset: {DATASET}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"JPEG-AI: {jpegai_dir}")
    print(f"Qualities: {qualities}")
    print(f"\nWARNING: 64×64 images require upscaling workaround!")
    print(f"         Upscale: 64×64 → 192×192 → compress → 64×64")
    print(f"         This causes ~88% accuracy ceiling.")

    if args.dry_run:
        print("\n[DRY RUN] No compression performed.")
        return

    if not jpegai_dir.exists():
        print(f"\nERROR: JPEG-AI directory not found: {jpegai_dir}")
        print("This script must be run on RunPod with JPEG-AI installed.")
        print("See: cloud-gpu/compress_eurosat.py for actual RunPod deployment.")
        return

    for qname in qualities:
        target_bpp = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"
{qname} (target_bpp={target_bpp/100:.2f})")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.tif"))
        for img_path in tqdm(images, desc=f"Compressing {qname}"):
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".png")
            compress_jpegai(img_path, out_path, target_bpp, jpegai_dir)

        print(f"  Compressed {len(images)} images to {out_dir}")

if __name__ == "__main__":
    main()
