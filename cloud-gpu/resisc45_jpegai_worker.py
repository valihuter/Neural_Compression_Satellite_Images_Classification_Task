#!/usr/bin/env python3
"""
JPEG-AI Compression Script for RESISC45 Dataset
Runs inside Docker container

Features:
- NO upscaling needed (256x256 > 128x128 minimum)
- Checkpoint support: Resume after interruption
- Progress tracking per quality level
- Skip already compressed images
- Parallel quality level support (run multiple containers)

Usage in Docker:
  python3 /scripts/compress_resisc45.py --quality q1,q2,q3
  python3 /scripts/compress_resisc45.py --quality q4,q5,q6
"""

import json
import subprocess
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
from datetime import datetime
import signal
import sys

# Paths (mounted in Docker)
INPUT_ROOT = Path('/data/input')
OUTPUT_ROOT = Path('/data/output')
JPEG_AI_ROOT = Path('/workspace/jpeg-ai-reference-software')
CHECKPOINT_FILE = OUTPUT_ROOT / 'checkpoint.json'

# Quality levels - adjusted for 256x256 images
# BPP values will be more accurate than EuroSAT (no upscaling distortion)
QUALITY_LEVELS = {
    'q1': 13,   # 0.13 BPP - lowest quality
    'q2': 25,   # 0.25 BPP
    'q3': 50,   # 0.50 BPP
    'q4': 75,   # 0.75 BPP
    'q5': 100,  # 1.00 BPP
    'q6': 150,  # 1.50 BPP - highest quality
}

# Global for signal handling
checkpoint_data = None

def save_checkpoint(data):
    """Save current progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

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
    output_path = OUTPUT_ROOT / f'jpeg_ai_{quality_name}' / img_path.parent.name / f'{img_path.stem}.png'
    return output_path.exists()

def compress_image(input_path, output_dir, target_bpp_x100):
    """
    Compress single 256x256 image with JPEG-AI
    
    RESISC45 images are 256x256 - well above JPEG-AI minimum of 128x128
    So we can compress directly without upscaling workaround!
    """
    import tempfile
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_path)
    final_output = output_dir / f'{input_path.stem}.png'
    
    temp_dir = Path(tempfile.mkdtemp())
    temp_input = temp_dir / "input.png"
    temp_output = temp_dir / "output.png"
    bitstream = temp_dir / "compressed.bin"
    
    try:
        # Convert to PNG for JPEG-AI (handles various input formats)
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Verify size
        if img.size != (256, 256):
            # Resize if needed (shouldn't happen for RESISC45)
            img = img.resize((256, 256), Image.BICUBIC)
        
        img.save(temp_input)
        
        # Encode with JPEG-AI
        encode_cmd = [
            'python3', '-m', 'src.reco.coders.encoder',
            str(temp_input), str(bitstream),
            '--set_target_bpp', str(target_bpp_x100),
            '--cfg', 'cfg/tools_on.json', 'cfg/profiles/base.json'
        ]
        
        result = subprocess.run(
            encode_cmd,
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per image
        )
        
        if result.returncode != 0:
            return False, f"Encode failed: {result.stderr[:200]}"
        
        # Decode
        decode_cmd = [
            'python3', '-m', 'src.reco.coders.decoder',
            str(bitstream), str(temp_output)
        ]
        
        result = subprocess.run(
            decode_cmd,
            cwd=JPEG_AI_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return False, f"Decode failed: {result.stderr[:200]}"
        
        # Copy output to final location
        shutil.copy2(temp_output, final_output)
        
        # Calculate actual BPP
        bpp = (bitstream.stat().st_size * 8) / (256 * 256)
        return True, bpp
        
    except subprocess.TimeoutExpired:
        return False, "Timeout expired"
    except Exception as e:
        return False, f"Exception: {str(e)[:200]}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_all_images():
    """Find all images in the input directory"""
    test_images = []
    
    for class_dir in sorted(INPUT_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        
        # Support various image formats
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            for img_path in sorted(class_dir.glob(ext)):
                test_images.append(img_path.relative_to(INPUT_ROOT))
    
    return test_images

def main():
    global checkpoint_data
    
    parser = argparse.ArgumentParser(description='JPEG-AI Compression for RESISC45')
    parser.add_argument('--quality', type=str, default='all',
                       help='Quality levels to process: "all" or comma-separated like "q1,q2,q3"')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of images (for testing)')
    args = parser.parse_args()
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 80)
    print("JPEG-AI Compression for RESISC45")
    print("=" * 80)
    print("Press Ctrl+C to stop - progress will be saved automatically")
    print("=" * 80)
    
    # Determine which quality levels to process
    if args.quality == 'all':
        quality_items = list(QUALITY_LEVELS.items())
    else:
        requested = [q.strip() for q in args.quality.split(',')]
        quality_items = [(q, QUALITY_LEVELS[q]) for q in requested if q in QUALITY_LEVELS]
    
    print(f"Quality levels: {[q[0] for q in quality_items]}")
    
    # Find all images
    test_images = get_all_images()
    
    if args.limit > 0:
        test_images = test_images[:args.limit]
        print(f"[Limited to {args.limit} images for testing]")
    
    print(f"Found {len(test_images)} images")
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    start_quality_idx = 0
    
    if checkpoint:
        print(f"\n[Resuming from checkpoint: {checkpoint['current_quality']}]")
        quality_names = [q[0] for q in quality_items]
        if checkpoint['current_quality'] in quality_names:
            start_quality_idx = quality_names.index(checkpoint['current_quality'])
    
    # Load or initialize results
    results_file = OUTPUT_ROOT / 'compression_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {
            'dataset': 'RESISC45',
            'image_size': '256x256',
            'start_time': datetime.now().isoformat(),
            'quality_levels': {},
            'errors': []
        }
    
    for idx, (qname, target_bpp) in enumerate(quality_items):
        # Skip already completed quality levels
        if idx < start_quality_idx:
            print(f"\n[Skipping {qname} - already completed]")
            continue
        
        print(f"\n{'='*80}")
        print(f"{qname}: Target {target_bpp/100:.2f} BPP")
        print(f"{'='*80}")
        
        quality_out = OUTPUT_ROOT / f'jpeg_ai_{qname}'
        success_count = 0
        skipped_count = 0
        error_count = 0
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
                error_count += 1
                results['errors'].append({
                    'quality': qname,
                    'image': str(img_path),
                    'error': result
                })
            
            # Save checkpoint every 100 images
            if (i + 1) % 100 == 0:
                save_checkpoint(checkpoint_data)
        
        elapsed = time.time() - start
        new_processed = success_count - skipped_count
        avg_bpp = total_bpp / new_processed if new_processed > 0 else 0
        
        results['quality_levels'][qname] = {
            'target_bpp': target_bpp / 100,
            'actual_bpp': round(avg_bpp, 4),
            'success': success_count,
            'skipped': skipped_count,
            'errors': error_count,
            'time_min': round(elapsed / 60, 2),
            'images_per_sec': round(len(test_images) / elapsed, 2) if elapsed > 0 else 0
        }
        
        # Save results after each quality level
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{success_count}/{len(test_images)} images done ({skipped_count} skipped, {error_count} errors)")
        print(f"Average BPP: {avg_bpp:.3f}")
        print(f"Time: {elapsed/60:.1f} minutes ({len(test_images)/elapsed:.1f} img/s)")
    
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
