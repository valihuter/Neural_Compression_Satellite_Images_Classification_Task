#!/usr/bin/env python3
"""
JPEG-AI Compression Script for EuroSAT Dataset
Runs inside Docker container with upscaling workaround

Features:
- Checkpoint support: Resume after interruption
- Progress tracking per quality level
- Skip already compressed images
"""

import json
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys

# Paths
INPUT_ROOT = Path('/data/input')
OUTPUT_ROOT = Path('/data/output')
JPEG_AI_ROOT = Path('/workspace/jpeg-ai-reference-software')
CHECKPOINT_FILE = OUTPUT_ROOT / 'checkpoint.json'

# Quality levels (BPP * 100)
QUALITY_LEVELS = {
    'q1': 13,   # 0.13 BPP
    'q2': 18,   # 0.18 BPP
    'q3': 26,   # 0.26 BPP
    'q4': 39,   # 0.39 BPP
    'q5': 53,   # 0.53 BPP
    'q6': 72,   # 0.72 BPP
}

# Global for signal handling
checkpoint_data = None

def save_checkpoint(data):
    """Save current progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[Checkpoint saved: {data['current_quality']}, {data['processed_count']} images]")

def load_checkpoint():
    """Load checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def signal_handler(sig, frame):
    """Handle Ctrl+C - save checkpoint before exit"""
    global checkpoint_data
    print("\n\nInterrupted! Saving checkpoint...")
    if checkpoint_data:
        save_checkpoint(checkpoint_data)
    print("You can resume later by running the script again.")
    sys.exit(0)

def is_already_compressed(img_path, quality_name):
    """Check if image was already compressed for this quality level"""
    output_path = OUTPUT_ROOT / f'jpeg_ai_{quality_name}' / img_path.parent.name / img_path.name
    return output_path.exists()

def compress_image(input_path, output_dir, target_bpp_x100):
    """Compress single image with 64x64 → 192x192 → 64x64 workaround"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)
    final_output = output_dir / input_path.name
    
    temp_dir = Path(tempfile.mkdtemp())
    upscaled_input = temp_dir / "input.png"
    upscaled_output = temp_dir / "output.png"
    bitstream = temp_dir / "compressed.bin"
    
    try:
        # Upscale 64→192
        img = Image.open(input_path)
        upscaled = img.resize((192, 192), Image.NEAREST)
        upscaled.save(upscaled_input)
        
        # Encode
        encode_cmd = [
            'python3', '-m', 'src.reco.coders.encoder',
            str(upscaled_input), str(bitstream),
            '--set_target_bpp', str(target_bpp_x100),
            '--cfg', 'cfg/tools_on.json', 'cfg/profiles/base.json'
        ]
        result = subprocess.run(
            encode_cmd,
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Encode failed: {result.stderr[:200]}"
        
        # Decode
        decode_cmd = [
            'python3', '-m', 'src.reco.coders.decoder',
            str(bitstream), str(upscaled_output)
        ]
        result = subprocess.run(
            decode_cmd,
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Decode failed: {result.stderr[:200]}"
        
        # Downscale 192→64
        decoded = Image.open(upscaled_output)
        downscaled = decoded.resize((64, 64), Image.BICUBIC)
        downscaled.save(final_output)
        
        # Calculate BPP
        bpp = (bitstream.stat().st_size * 8) / (64 * 64)
        return True, bpp
        
    except Exception as e:
        return False, f"Exception: {str(e)[:200]}"
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    global checkpoint_data
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("JPEG-AI Compression for EuroSAT")
    
    # Find all images (supports .jpg, .jpeg, .tif, .png)
    test_images = []
    for class_dir in INPUT_ROOT.iterdir():
        if class_dir.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.tif', '*.png']:
                for img_path in class_dir.glob(ext):
                    test_images.append(img_path.relative_to(INPUT_ROOT))
    
    test_images = sorted(test_images)  # Sort for consistent ordering
    
    print(f"Found {len(test_images)} images")
    print(f"Quality levels: {list(QUALITY_LEVELS.keys())}")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    start_quality_idx = 0
    
    if checkpoint:
        print(f"\n[Resuming from checkpoint: {checkpoint['current_quality']}]")
        # Find which quality level to resume from
        quality_names = list(QUALITY_LEVELS.keys())
        if checkpoint['current_quality'] in quality_names:
            start_quality_idx = quality_names.index(checkpoint['current_quality'])
    
    # Load or initialize results
    results_file = OUTPUT_ROOT / 'compression_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            'start_time': datetime.now().isoformat(),
            'quality_levels': {},
            'errors': []
        }
    
    quality_items = list(QUALITY_LEVELS.items())
    
    for idx, (qname, target_bpp) in enumerate(quality_items):
        # Skip already completed quality levels
        if idx < start_quality_idx:
            print(f"\n[Skipping {qname} - already completed]")
            continue
        
        print(f"{qname}: Target {target_bpp/100:.2f} BPP")
        
        quality_out = OUTPUT_ROOT / f'jpeg_ai_{qname}'
        success_count = 0
        skipped_count = 0
        total_bpp = 0
        start = time.time()
        
        for i, img_path in enumerate(tqdm(test_images, desc=qname)):
            # Update checkpoint data
            checkpoint_data = {
                'current_quality': qname,
                'current_image': str(img_path),
                'processed_count': i,
                'total_images': len(test_images),
                'timestamp': datetime.now().isoformat()
            }
            
            # Skip if already compressed
            if is_already_compressed(img_path, qname):
                skipped_count += 1
                success_count += 1
                continue
            
            input_full = INPUT_ROOT / img_path
            class_name = img_path.parent.name
            output_dir = quality_out / class_name
            
            success, result = compress_image(input_full, output_dir, target_bpp)
            
            if success:
                success_count += 1
                total_bpp += result
            else:
                results['errors'].append({
                    'quality': qname,
                    'image': str(img_path),
                    'error': result
                })
            
            # Save checkpoint every 100 images
            if (i + 1) % 100 == 0:
                save_checkpoint(checkpoint_data)
        
        elapsed = time.time() - start
        avg_bpp = total_bpp / (success_count - skipped_count) if (success_count - skipped_count) > 0 else 0
        
        results['quality_levels'][qname] = {
            'target_bpp': target_bpp / 100,
            'actual_bpp': avg_bpp,
            'success': success_count,
            'skipped': skipped_count,
            'errors': len(test_images) - success_count,
            'time_min': elapsed / 60
        }
        
        # Save results after each quality level
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{success_count} images done ({skipped_count} skipped)")
        print(f"Average BPP: {avg_bpp:.3f}")
        print(f"Time: {elapsed/60:.1f} minutes")
    
    results['end_time'] = datetime.now().isoformat()
    
    # Save final results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Remove checkpoint file on successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    print("COMPLETE")
    print(f"Results: {results_file}")
    print(f"Total errors: {len(results['errors'])}")

if __name__ == '__main__':
    main()
