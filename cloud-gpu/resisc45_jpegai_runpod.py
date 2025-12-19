#!/usr/bin/env python3
"""
JPEG-AI Parallel Compression Script for RESISC45 - Pod 5 (CPU)
Verarbeitet 4 Klassen parallel: circular_farmland, dense_residential, freeway, industrial_area
"""

import json
import subprocess
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Paths for RunPod
INPUT_ROOT = Path("/workspace/data/NWPU-RESISC45")
OUTPUT_ROOT = Path("/workspace/output/resisc45_jpegai")
JPEG_AI_ROOT = Path("/workspace/jpeg-ai-reference-software")
CHECKPOINT_FILE = OUTPUT_ROOT / "checkpoint_pod5_parallel.json"

# Number of parallel workers (16 CPUs Pod 5, 8 Worker für gute Balance)
NUM_WORKERS = 8

# Quality levels
QUALITY_LEVELS = {
    "q1": 13, "q2": 25, "q3": 50, "q4": 75, "q5": 100, "q6": 150,
}

# Zu bearbeitende Klassen (die restlichen EuroSAT-überlappenden)
CLASSES_FILTER = [
    "circular_farmland", "dense_residential", "freeway", "industrial_area"
]

def compress_image(args):
    """Compress a single image - runs in worker process"""
    input_path, output_dir, target_bpp_x100 = args
    import tempfile
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)
    final_output = output_dir / f"{input_path.stem}.png"
    
    # Skip if already exists
    if final_output.exists():
        return True, 0, "skipped"
    
    temp_dir = Path(tempfile.mkdtemp())
    temp_input = temp_dir / "input.png"
    temp_output = temp_dir / "output.png"
    bitstream = temp_dir / "compressed.bin"
    
    venv_python = str(JPEG_AI_ROOT / "venv" / "bin" / "python")
    
    try:
        # Convert to PNG (JPEG-AI only accepts PNG!)
        img = Image.open(input_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(temp_input)
        
        # Encode
        encode_cmd = [
            venv_python, "-m", "src.reco.coders.encoder",
            str(temp_input), str(bitstream),
            "--set_target_bpp", str(target_bpp_x100),
            "--cfg", "cfg/tools_on.json", "cfg/profiles/base.json",
            "-target_device", "cpu"
        ]
        
        result = subprocess.run(
            encode_cmd,
            cwd=str(JPEG_AI_ROOT),
            capture_output=True,
            timeout=300
        )
        
        if result.returncode != 0 or not bitstream.exists():
            return False, 0, f"Encode failed: {result.stderr.decode()[:100]}"
        
        # Decode
        decode_cmd = [
            venv_python, "-m", "src.reco.coders.decoder",
            str(bitstream), str(temp_output),
            "-target_device", "cpu"
        ]
        
        result = subprocess.run(
            decode_cmd,
            cwd=str(JPEG_AI_ROOT),
            capture_output=True,
            timeout=300
        )
        
        if result.returncode != 0 or not temp_output.exists():
            return False, 0, f"Decode failed: {result.stderr.decode()[:100]}"
        
        # Copy to final location
        shutil.copy2(temp_output, final_output)
        
        # Calculate BPP
        bpp = (bitstream.stat().st_size * 8) / (256 * 256)
        return True, bpp, "compressed"
        
    except subprocess.TimeoutExpired:
        return False, 0, "Timeout"
    except Exception as e:
        return False, 0, str(e)[:100]
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_images_to_process():
    """Get all images from filtered classes (only .jpg, no ._ files!)"""
    all_images = []
    
    for class_dir in sorted(INPUT_ROOT.iterdir()):
        if class_dir.is_dir() and class_dir.name in CLASSES_FILTER:
            for img_path in sorted(class_dir.glob("*.jpg")):
                # Skip MacOS resource fork files
                if not img_path.name.startswith("._"):
                    all_images.append(img_path)
    
    return all_images

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    all_images = get_images_to_process()
    
    print("=" * 80)
    print("JPEG-AI PARALLEL Compression for RESISC45 - Pod 5")
    print("=" * 80)
    print(f"Classes: {CLASSES_FILTER}")
    print(f"Quality levels: {list(QUALITY_LEVELS.keys())}")
    print(f"Found {len(all_images)} images ({len(CLASSES_FILTER)} classes × 700)")
    print(f"Parallel workers: {NUM_WORKERS}")
    print("=" * 80)
    
    for qname, target_bpp in QUALITY_LEVELS.items():
        quality_dir = OUTPUT_ROOT / f"jpeg_ai_{qname}"
        quality_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"{qname}: Target {target_bpp/100:.2f} BPP (parallel)")
        print(f"{'='*80}")
        
        # Prepare tasks
        tasks = []
        for img_path in all_images:
            class_name = img_path.parent.name
            output_dir = quality_dir / class_name
            output_file = output_dir / f"{img_path.stem}.png"
            
            # Skip if already exists
            if not output_file.exists():
                tasks.append((str(img_path), str(output_dir), target_bpp))
        
        already_done = len(all_images) - len(tasks)
        print(f"Already done: {already_done}, To process: {len(tasks)}")
        
        if not tasks:
            print(f"{qname}: All images already compressed!")
            continue
        
        # Process in parallel
        success = 0
        errors = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(compress_image, task): task for task in tasks}
            
            pbar = tqdm(as_completed(futures), total=len(tasks), desc=qname)
            for future in pbar:
                try:
                    ok, bpp, status = future.result()
                    if ok:
                        success += 1
                    else:
                        errors += 1
                        if errors <= 3:
                            print(f"\nError: {status}")
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"\nException: {e}")
                
                pbar.set_postfix({
                    "done": success + already_done,
                    "err": errors,
                    "total": len(all_images)
                })
        
        elapsed = time.time() - start_time
        print(f"\n{qname} done: {success} new, {already_done} skipped, {errors} errors")
        print(f"Time: {elapsed/60:.1f} min ({elapsed/max(success,1):.1f}s/img effective)")
        print(f"Throughput: {success/(elapsed/60):.1f} img/min with {NUM_WORKERS} workers")
    
    print("COMPLETE")


if __name__ == "__main__":
    # Use spawn to avoid issues with CUDA/torch
    multiprocessing.set_start_method('spawn', force=True)
    main()
