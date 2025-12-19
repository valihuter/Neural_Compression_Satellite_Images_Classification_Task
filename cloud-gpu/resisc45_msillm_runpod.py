#!/usr/bin/env python3
"""
MS-ILLM Compression for RESISC45 - Multi-GPU Optimized

Uses both GPUs efficiently by assigning each worker to a specific GPU.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path("/resisc45")
OUTPUT_ROOT = Path("/workspace/output/resisc45_msillm")
CHECKPOINT_FILE = OUTPUT_ROOT / "checkpoint_msillm.json"

# Multi-GPU setup - 2x RTX 4090
NUM_GPUS = 2
BATCH_SIZE = 4  # Images per GPU batch

# MS-ILLM quality levels
QUALITY_LEVELS = {
    'q1': {'model': 'msillm_quality_1', 'target_bpp': 0.035},
    'q2': {'model': 'msillm_quality_2', 'target_bpp': 0.07},
    'q3': {'model': 'msillm_quality_3', 'target_bpp': 0.14},
    'q4': {'model': 'msillm_quality_4', 'target_bpp': 0.30},
    'q5': {'model': 'msillm_quality_5', 'target_bpp': 0.45},
    'q6': {'model': 'msillm_quality_6', 'target_bpp': 0.90}
}

# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {
        "completed_qualities": [],
        "current_quality": None,
        "processed_images": {},
        "start_time": datetime.now().isoformat()
    }

def save_checkpoint(checkpoint):
    checkpoint["last_update"] = datetime.now().isoformat()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# ============================================================
# GPU WORKER FUNCTION
# ============================================================

def init_worker(gpu_id):
    """Initialize worker with specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    global worker_gpu_id
    worker_gpu_id = gpu_id

def compress_batch_on_gpu(args):
    """
    Compress a batch of images on a specific GPU.
    
    Args:
        args: (gpu_id, image_list, model_name, quality_output_dir)
    """
    gpu_id, image_list, model_name, quality_output_dir = args
    
    # Use GPU directly by index
    device = f'cuda:{gpu_id}'
    
    results = []
    
    try:
        # Load model once for entire batch
        model = torch.hub.load(
            "facebookresearch/NeuralCompression",
            model_name,
            verbose=False
        )
        model = model.to(device)
        model.eval()
        
        transform = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        for img_path in image_list:
            img_path = Path(img_path)
            output_class_dir = Path(quality_output_dir) / img_path.parent.name
            output_file = output_class_dir / f"{img_path.stem}.png"
            
            # Skip if exists
            if output_file.exists():
                results.append((True, 0, "skipped"))
                continue
            
            try:
                # Load and process image
                img = Image.open(img_path).convert('RGB')
                width, height = img.size
                
                x = transform(img).unsqueeze(0).to(device)
                
                # Compress
                with torch.no_grad():
                    out = model(x)
                    if isinstance(out, dict):
                        x_hat = out.get('x_hat', out.get('reconstruction'))
                    elif isinstance(out, tuple):
                        x_hat = out[0]
                    else:
                        x_hat = out
                
                # Save
                output_file.parent.mkdir(parents=True, exist_ok=True)
                x_hat_img = to_pil(x_hat.squeeze(0).clamp(0, 1).cpu())
                x_hat_img.save(output_file, 'PNG')
                
                # Calculate BPP
                file_size = output_file.stat().st_size
                bpp = (file_size * 8) / (width * height)
                
                results.append((True, bpp, "success"))
                
            except Exception as e:
                results.append((False, 0, str(e)[:100]))
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        # Model loading failed
        for _ in image_list:
            results.append((False, 0, f"Model error: {str(e)[:100]}"))
    
    return results

# ============================================================
# MAIN PROCESSING
# ============================================================

def process_quality_level(quality_name, quality_params, checkpoint):
    """Process all images for one quality level using both GPUs"""
    
    model_name = quality_params['model']
    target_bpp = quality_params['target_bpp']
    
    print(f"\n{'='*60}")
    print(f"Quality: {quality_name} ({model_name})")
    print(f"Target BPP: {target_bpp}")
    print(f"{'='*60}")
    
    # Find all images
    all_images = []
    for class_dir in INPUT_ROOT.iterdir():
        if class_dir.is_dir():
            for img_path in class_dir.glob("*.jpg"):
                if not img_path.name.startswith('._'):
                    all_images.append(img_path)
    
    all_images = sorted(all_images)
    print(f"Total images: {len(all_images)}")
    
    # Check already processed
    quality_output = OUTPUT_ROOT / quality_name
    
    # Build task list (only unprocessed)
    tasks = []
    for img_path in all_images:
        output_class_dir = quality_output / img_path.parent.name
        output_file = output_class_dir / f"{img_path.stem}.png"
        if not output_file.exists():
            tasks.append(str(img_path))
    
    already_done = len(all_images) - len(tasks)
    print(f"Already done: {already_done}")
    print(f"To process: {len(tasks)}")
    
    if not tasks:
        print("All images already processed!")
        return
    
    # Split tasks into batches for each GPU
    batch_size = BATCH_SIZE
    batches = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        gpu_id = (len(batches)) % NUM_GPUS  # Alternate between GPUs
        batches.append((gpu_id, batch, model_name, str(quality_output)))
    
    print(f"Processing {len(batches)} batches across {NUM_GPUS} GPUs...")
    
    # Process with multiprocessing
    start_time = time.time()
    success_count = 0
    error_count = 0
    total_bpp = 0
    
    # Use spawn to avoid CUDA issues
    ctx = torch.multiprocessing.get_context('spawn')
    
    with tqdm(total=len(tasks), desc=quality_name) as pbar:
        with ProcessPoolExecutor(max_workers=NUM_GPUS, mp_context=ctx) as executor:
            futures = {executor.submit(compress_batch_on_gpu, batch): batch for batch in batches}
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    for success, bpp, status in results:
                        if success:
                            if status == "success":
                                success_count += 1
                                total_bpp += bpp
                        else:
                            error_count += 1
                        pbar.update(1)
                except Exception as e:
                    print(f"\nBatch error: {e}")
                    error_count += BATCH_SIZE
                    pbar.update(BATCH_SIZE)
    
    elapsed = time.time() - start_time
    avg_bpp = total_bpp / success_count if success_count > 0 else 0
    
    print(f"\n {quality_name} complete:")
    print(f"  Success: {success_count}, Errors: {error_count}")
    print(f"  Avg BPP: {avg_bpp:.3f} (target: {target_bpp})")
    print(f"  Time: {elapsed/60:.1f} min ({len(tasks)/elapsed:.2f} img/s)")
    
    # Update checkpoint
    checkpoint["completed_qualities"].append(quality_name)
    checkpoint["processed_images"][quality_name] = success_count + already_done
    save_checkpoint(checkpoint)

def main():
    print("="*60)
    print("MS-ILLM Compression - Multi-GPU Optimized")
    print("="*60)
    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Verify paths
    if not INPUT_ROOT.exists():
        print(f"\n ERROR: Input path not found: {INPUT_ROOT}")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n ERROR: CUDA not available!")
        sys.exit(1)
    
    print(f"\nAvailable GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = checkpoint.get("completed_qualities", [])
    print(f"\nCheckpoint: {len(completed)} qualities completed")
    
    # Process each quality level
    for quality_name, quality_params in QUALITY_LEVELS.items():
        if quality_name in completed:
            print(f"\n  Skipping {quality_name} (already complete)")
            continue
        
        checkpoint["current_quality"] = quality_name
        save_checkpoint(checkpoint)
        
        process_quality_level(quality_name, quality_params, checkpoint)
    
    print("COMPLETE")


if __name__ == "__main__":
    # Required for multiprocessing with CUDA
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
