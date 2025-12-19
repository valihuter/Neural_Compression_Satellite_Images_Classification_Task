#!/usr/bin/env python3
"""
JPEG-AI Compression Script for Cloud-GPU
=========================================

Komprimiert EuroSAT-Bilder mit JPEG-AI auf 6 Quality-Levels.
Optimiert für Cloud-GPU mit Checkpoint-Support.

Usage:
    python compress_eurosat.py

Voraussetzungen:
    - JPEG-AI setup via setup_and_run.sh
    - EuroSAT_RGB entpackt in /workspace/EuroSAT_RGB/
    - rgb_test.json in /workspace/
"""

import os
import sys
import json
import time
import shutil
import signal
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
WORKSPACE = Path("/workspace")
JPEGAI_DIR = WORKSPACE / "jpeg-ai-reference-software"
EUROSAT_PATH = WORKSPACE / "EuroSAT_RGB"
SPLIT_FILE = WORKSPACE / "rgb_test.json"
OUTPUT_ROOT = WORKSPACE / "results" / "jpeg_ai"
CHECKPOINT_FILE = OUTPUT_ROOT / "checkpoint.json"

# Quality levels (target_bpp * 100)
QUALITY_LEVELS = {
    'q1': 13,   # 0.13 BPP - highest compression
    'q2': 25,   # 0.25 BPP
    'q3': 35,   # 0.35 BPP
    'q4': 45,   # 0.45 BPP
    'q5': 55,   # 0.55 BPP
    'q6': 72,   # 0.72 BPP - best quality
}

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# COMPRESSION FUNCTION
# =============================================================================

def compress_single_image(input_path: Path, output_path: Path, target_bpp_x100: int) -> tuple:
    """
    Compress a single 64x64 EuroSAT image with JPEG-AI.
    
    Workaround: JPEG-AI requires min 160x160 due to eICCI filter,
    so we upscale to 192x192, compress, then downscale back.
    
    Returns:
        (success: bool, bpp_or_error: float|str)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path(tempfile.mkdtemp())
    upscaled_input = temp_dir / "input_192.png"
    bitstream = temp_dir / "compressed.bin"
    upscaled_output = temp_dir / "output_192.png"
    
    try:
        # Step 1: Upscale 64x64 → 192x192
        img = Image.open(input_path)
        if img.size != (64, 64):
            img = img.resize((64, 64), Image.BICUBIC)
        upscaled = img.resize((192, 192), Image.NEAREST)
        upscaled.save(upscaled_input)
        
        # Step 2: Encode (with GPU)
        encode_cmd = [
            'python', '-m', 'src.reco.coders.encoder',
            str(upscaled_input), str(bitstream),
            '--set_target_bpp', str(target_bpp_x100),
            '-target_device', 'gpu',
            '--cfg', 'cfg/tools_on.json', 'cfg/profiles/base.json'
        ]
        result = subprocess.run(encode_cmd, capture_output=True, cwd=str(JPEGAI_DIR), timeout=60)
        if result.returncode != 0:
            return False, f"Encode error: {result.stderr.decode()[:200]}"
        
        # Step 3: Decode (with GPU)
        decode_cmd = [
            'python', '-m', 'src.reco.coders.decoder',
            str(bitstream), str(upscaled_output),
            '-target_device', 'gpu'
        ]
        result = subprocess.run(decode_cmd, capture_output=True, cwd=str(JPEGAI_DIR), timeout=60)
        if result.returncode != 0:
            return False, f"Decode error: {result.stderr.decode()[:200]}"
        
        # Step 4: Downscale 192x192 → 64x64
        decoded = Image.open(upscaled_output)
        downscaled = decoded.resize((64, 64), Image.BICUBIC)
        downscaled.save(output_path)
        
        # Calculate BPP based on original 64x64 size
        bpp = (bitstream.stat().st_size * 8) / (64 * 64)
        
        return True, bpp
        
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    global shutdown_requested
    
    # Verify paths
    if not EUROSAT_PATH.exists():
        print(f"Error: EuroSAT not found at {EUROSAT_PATH}")
        print("Please upload and unzip EuroSAT_RGB.zip")
        sys.exit(1)
    
    if not SPLIT_FILE.exists():
        print(f"Error: Split file not found at {SPLIT_FILE}")
        print("Please upload rgb_test.json")
        sys.exit(1)
    
    # Load test split
    with open(SPLIT_FILE) as f:
        test_split = json.load(f)
    
    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load or create checkpoint
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {
            'completed_qualities': [],
            'current_quality': None,
            'current_index': 0,
            'results': {},
            'errors': []
        }
    
    # Process each quality level
    for qname, target_bpp in QUALITY_LEVELS.items():
        if shutdown_requested:
            break
            
        # Skip completed
        if qname in checkpoint['completed_qualities']:
            continue
        
        quality_out = OUTPUT_ROOT / f'jpeg_ai_{qname}'
        
        # Resume from checkpoint
        start_idx = 0
        if checkpoint['current_quality'] == qname:
            start_idx = checkpoint['current_index']
        
        success_count = 0
        total_bpp = 0.0
        start_time = time.time()
        
        pbar = tqdm(test_split[start_idx:], initial=start_idx, total=len(test_split), desc=qname)
        
        for idx, img_path in enumerate(pbar):
            if shutdown_requested:
                # Save checkpoint before exit
                checkpoint['current_quality'] = qname
                checkpoint['current_index'] = start_idx + idx
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                break
            
            actual_idx = start_idx + idx
            
            input_full = EUROSAT_PATH / img_path
            class_name = Path(img_path).parent.name
            output_dir = quality_out / class_name
            output_file = output_dir / Path(img_path).name
            
            # Skip existing
            if output_file.exists():
                success_count += 1
                continue
            
            success, result = compress_single_image(input_full, output_file, target_bpp)
            
            if success:
                success_count += 1
                total_bpp += result
                pbar.set_postfix({'bpp': f'{result:.3f}', 'ok': success_count})
            else:
                checkpoint['errors'].append({
                    'quality': qname,
                    'image': str(img_path),
                    'error': str(result)[:200]
                })
            
            # Checkpoint every 500 images
            if (actual_idx + 1) % 500 == 0:
                checkpoint['current_quality'] = qname
                checkpoint['current_index'] = actual_idx + 1
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)
        
        if shutdown_requested:
            break
        
        # Quality level complete
        elapsed = time.time() - start_time
        avg_bpp = total_bpp / max(success_count, 1)
        
        checkpoint['results'][qname] = {
            'target_bpp': target_bpp / 100,
            'actual_bpp': avg_bpp,
            'success': success_count,
            'total': len(test_split),
            'time_minutes': elapsed / 60
        }
        checkpoint['completed_qualities'].append(qname)
        checkpoint['current_quality'] = None
        checkpoint['current_index'] = 0
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
    # Final summary
    if not shutdown_requested:
        # Save final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'results': checkpoint['results'],
            'errors': checkpoint['errors']
        }
        
        with open(OUTPUT_ROOT / 'compression_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create ZIP for download
        shutil.make_archive(
            str(WORKSPACE / 'jpeg_ai_results'),
            'zip',
            OUTPUT_ROOT.parent,
            OUTPUT_ROOT.name
        )
        
        print(f"Results saved to: {WORKSPACE / 'jpeg_ai_results.zip'}")
        print(f"Errors: {len(checkpoint['errors'])}")
        
        print(f"{'Quality':<8} {'Target':<10} {'Actual':<10} {'Images':<10} {'Time':<10}")
        for q, stats in checkpoint['results'].items():
            print(f"{q:<8} {stats['target_bpp']:.2f} BPP   {stats['actual_bpp']:.3f} BPP  {stats['success']:<10} {stats['time_minutes']:.1f} min")
    else:
        print("Compression interrupted. Run again to resume.")

if __name__ == '__main__':
    main()
