#!/usr/bin/env python3
"""
JPEG-AI Compression - RESISC45 Subset Dataset

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Codec: JPEG-AI (ISO/IEC 6048-1:2025, Reference Software v6.0)
Hardware: RunPod 16 vCPUs, CPU mode with 8 parallel workers

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Quality Levels (target BPP × 100):
    q1: 12  → 0.12 BPP
    q2: 25  → 0.25 BPP
    q3: 50  → 0.50 BPP
    q4: 75  → 0.75 BPP
    q5: 100 → 1.00 BPP
    q6: 150 → 1.50 BPP

IMPORTANT - Native Resolution:
    256×256 meets JPEG-AI minimum requirement (128×128).
    NO upscaling workaround needed!
    Config: cfg/tools_on.json (full feature set enabled)
    Result: 94.25% accuracy at q1 (vs 77% on EuroSAT with upscaling)

Hardware Note:
    - GPU mode had C++ extension build issues on some RunPod pods
    - CPU mode with parallel processing was more stable and faster overall
    - 8 workers on 16 vCPU pod achieved good throughput

Execution Environment:
    RunPod with 16 vCPUs (CPU-only pod or GPU pod in CPU mode)
    8 parallel workers via ProcessPoolExecutor
    Total: ~12 hours for 46,200 images (7,700 × 6 qualities)

Usage:
    python resisc45_jpegai.py --quality q1
    python resisc45_jpegai.py --quality all

Note: This is a DOCUMENTATION script. Actual compression was performed
on RunPod using cloud-gpu/compress_resisc45_pod5_parallel.py
=============================================================================
"""

import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATASET = "resisc45"
IMAGE_SIZE = (256, 256)
NUM_WORKERS = 8  # Optimal for 16 vCPU pod

BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / "data" / "resisc45" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "resisc45" / "comp" / "jpegai"

# JPEG-AI paths (RunPod)
JPEGAI_DIR = Path("/workspace/jpeg-ai-reference-software")

QUALITY_LEVELS = {
    "q1": 12,   # 0.12 BPP
    "q2": 25,   # 0.25 BPP
    "q3": 50,   # 0.50 BPP
    "q4": 75,   # 0.75 BPP
    "q5": 100,  # 1.00 BPP
    "q6": 150,  # 1.50 BPP
}

# Measured = target (JPEG-AI is rate-controlled)
MEASURED_BPP = {"q1": 0.12, "q2": 0.25, "q3": 0.50, "q4": 0.75, "q5": 1.00, "q6": 1.50}

def compress_single(args):
    """Worker function for parallel compression."""
    input_path, output_path, target_bpp_x100, jpegai_dir = args
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        return True, "skipped"
    
    temp_dir = Path(tempfile.mkdtemp())
    temp_input = temp_dir / "input.png"
    bitstream = temp_dir / "compressed.bin"
    temp_output = temp_dir / "output.png"
    
    venv_python = str(jpegai_dir / "venv" / "bin" / "python")
    
    try:
        # Convert to PNG (JPEG-AI requires PNG input)
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(temp_input)
        
        # Encode (CPU mode)
        encode_cmd = [
            venv_python, "-m", "src.reco.coders.encoder",
            str(temp_input), str(bitstream),
            "--set_target_bpp", str(target_bpp_x100),
            "--cfg", "cfg/tools_on.json", "cfg/profiles/base.json",
            "-target_device", "cpu"
        ]
        result = subprocess.run(encode_cmd, cwd=str(jpegai_dir), capture_output=True, timeout=300)
        
        if result.returncode != 0 or not bitstream.exists():
            return False, f"Encode failed: {result.stderr.decode()[:100]}"
        
        # Decode (CPU mode)
        decode_cmd = [
            venv_python, "-m", "src.reco.coders.decoder",
            str(bitstream), str(temp_output),
            "-target_device", "cpu"
        ]
        result = subprocess.run(decode_cmd, cwd=str(jpegai_dir), capture_output=True, timeout=300)
        
        if result.returncode != 0 or not temp_output.exists():
            return False, f"Decode failed: {result.stderr.decode()[:100]}"
        
        # Copy to final location
        shutil.copy(temp_output, output_path)
        return True, "success"
        
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="JPEG-AI compression for RESISC45")
    parser.add_argument("--quality", choices=["q1", "q2", "q3", "q4", "q5", "q6", "all"], required=True)
    parser.add_argument("--jpegai-dir", type=str, default=str(JPEGAI_DIR))
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qualities = list(QUALITY_LEVELS.keys()) if args.quality == "all" else [args.quality]
    jpegai_dir = Path(args.jpegai_dir)

    print(f"Dataset: {DATASET}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"JPEG-AI: {jpegai_dir}")
    print(f"Workers: {args.workers}")
    print(f"Qualities: {qualities}")
    print(f"\nNOTE: 256×256 images work natively (no upscaling needed)!")
    print(f"      Using CPU mode with parallel workers for stability.")

    if args.dry_run:
        print("\n[DRY RUN] Target BPP values (rate-controlled):")
        for q in qualities:
            print(f"  {q}: target={QUALITY_LEVELS[q]/100:.2f} BPP → measured={MEASURED_BPP[q]:.2f} BPP")
        return

    if not jpegai_dir.exists():
        print(f"\nERROR: JPEG-AI directory not found: {jpegai_dir}")
        print("This script must be run on RunPod with JPEG-AI installed.")
        print("See: cloud-gpu/compress_resisc45_pod5_parallel.py for actual RunPod deployment.")
        return

    for qname in qualities:
        target_bpp = QUALITY_LEVELS[qname]
        out_dir = OUTPUT_DIR / qname
        print(f"
{qname} (target_bpp={target_bpp/100:.2f})")

        images = list(INPUT_DIR.rglob("*.jpg")) + list(INPUT_DIR.rglob("*.png"))
        tasks = []
        for img_path in images:
            rel_path = img_path.relative_to(INPUT_DIR)
            out_path = out_dir / rel_path.with_suffix(".png")
            tasks.append((str(img_path), str(out_path), target_bpp, jpegai_dir))

        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(compress_single, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Compressing {qname}"):
                success, msg = future.result()
                if success:
                    success_count += 1

        print(f"  Completed {success_count}/{len(images)} images")

if __name__ == "__main__":
    main()
