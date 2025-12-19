#!/usr/bin/env python3
"""
Labelling Tool for Miscompression Analysis

This script allows interactive labelling of miscompressed images using VPV.
It:
1. Asks user to select codec (quality fixed to q6, models combined)
2. Creates temporary folders with original and compressed images
3. Generates a CSV with image metadata (filename, quality, codec, predictions)
4. Opens VPV for visual comparison and annotation
5. Captures all TAGs from VPV output and saves them to a label file
"""

import os
import sys
import shutil
import subprocess
import random
import pandas as pd
import pty
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UNCOMP_DIR = DATA_DIR / "uncomp" / "EuroSAT_RGB"
COMPRESSED_DIR = DATA_DIR / "compressed_neural"
MISCLASSIFICATIONS_CSV = BASE_DIR / "results" / "miscompression_analysis" / "misclassifications.csv"
LABELLING_DIR = BASE_DIR / "labelling"

# VPV executable path (use local build with tagging support)
VPV_PATH = Path.home() / "lokalUIBK" / "seclab-vpv-clone" / "build" / "vpv.app" / "Contents" / "MacOS" / "vpv"

# Available options
CODECS = ["cheng2020-attn", "msillm", "jpeg_ai"]
QUALITY = "q6"  # Fixed to q6 for review

# Maximum images per codec for labelling (random sample if more available)
# msillm has only 98 unique images at q6, so we sample 90 from each codec
# for balanced comparison across codecs
MAX_IMAGES_PER_CODEC = 90

# Random seed for reproducibility
# Using seed 42 ensures the same 90 images are selected on every run,
# allowing consistent labelling sessions and reproducible results
RANDOM_SEED = 42

# Images to exclude (misclassified in ALL 6 configurations - likely dataset edge cases)
EXCLUDED_IMAGES = [
    "Forest/Forest_749.jpg",
    "PermanentCrop/PermanentCrop_1584.jpg",
    "PermanentCrop/PermanentCrop_34.jpg",
    "PermanentCrop/PermanentCrop_661.jpg",
    "River/River_1327.jpg",
]

def clear_temp_folders():
    """Remove and recreate temporary folders."""
    temp_uncomp = LABELLING_DIR / "temp_uncomp"
    temp_comp = LABELLING_DIR / "temp_comp"
    
    if temp_uncomp.exists():
        shutil.rmtree(temp_uncomp)
    if temp_comp.exists():
        shutil.rmtree(temp_comp)
    
    temp_uncomp.mkdir(parents=True, exist_ok=True)
    temp_comp.mkdir(parents=True, exist_ok=True)
    
    return temp_uncomp, temp_comp

def get_user_selection():
    """Get user input for codec (model is not used - we combine both)."""
    
    # Select Codec
    print("\nAvailable codecs:")
    for i, c in enumerate(CODECS, 1):
        print(f"  {i}. {c}")
    while True:
        try:
            c_idx = int(input("\nSelect codec (1-3): ")) - 1
            if 0 <= c_idx < len(CODECS):
                codec = CODECS[c_idx]
                break
            print("Invalid selection. Please enter 1-3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    return codec

def load_misclassifications(quality, codec):
    """Load misclassifications for the selected codec (all models combined, unique images)."""
    if not MISCLASSIFICATIONS_CSV.exists():
        print(f"Error: Misclassifications CSV not found at {MISCLASSIFICATIONS_CSV}")
        sys.exit(1)
    
    df = pd.read_csv(MISCLASSIFICATIONS_CSV)
    
    # Filter for selected quality and codec (combine both models)
    filtered = df[
        (df['quality'] == quality) & 
        (df['codec'] == codec)
    ].copy()
    
    # Exclude images that fail in all 6 configurations
    filtered = filtered[~filtered['filename'].isin(EXCLUDED_IMAGES)]
    
    # Get unique images (deduplicate across models)
    # Keep first occurrence with aggregated model info
    unique_df = filtered.groupby('filename').agg({
        'true_class': 'first',
        'original_pred': 'first',
        'compressed_pred': lambda x: list(x.unique()),  # may differ by model
        'original_confidence': 'max',
        'compressed_confidence': 'max',
        'model': lambda x: list(x.unique())  # which models were affected
    }).reset_index()
    
    total_available = len(unique_df)
    print(f"\nFound {total_available} unique images for {codec} / {quality}")
    print(f"  (Excluded {len(EXCLUDED_IMAGES)} images that fail in all configurations)")
    
    # Sample randomly if more than MAX_IMAGES_PER_CODEC
    if len(unique_df) > MAX_IMAGES_PER_CODEC:
        random.seed(RANDOM_SEED)
        unique_df = unique_df.sample(n=MAX_IMAGES_PER_CODEC, random_state=RANDOM_SEED)
        print(f"  → Randomly sampled {MAX_IMAGES_PER_CODEC} images for labelling")
    else:
        print(f"  → Using all {len(unique_df)} images (less than {MAX_IMAGES_PER_CODEC})")
    
    return unique_df

def prepare_images(misclassifications_df, quality, codec):
    """Copy images to temporary folders for VPV viewing."""
    temp_uncomp, temp_comp = clear_temp_folders()
    
    # Get compressed image folder path
    codec_quality_folder = f"{codec}_{quality}"
    comp_base = COMPRESSED_DIR / codec / codec_quality_folder
    
    if not comp_base.exists():
        print(f"Error: Compressed folder not found at {comp_base}")
        sys.exit(1)
    
    copied_images = []
    
    for idx, row in misclassifications_df.iterrows():
        filename = row['filename']  # e.g., "Forest/Forest_2243.jpg"
        
        # Parse filename
        parts = filename.split('/')
        class_name = parts[0]
        base_name = parts[1]  # e.g., "Forest_2243.jpg"
        name_without_ext = os.path.splitext(base_name)[0]  # e.g., "Forest_2243"
        
        # Source paths
        original_path = UNCOMP_DIR / class_name / base_name
        compressed_path = comp_base / class_name / f"{name_without_ext}.png"
        
        if not original_path.exists():
            print(f"Warning: Original not found: {original_path}")
            continue
        if not compressed_path.exists():
            print(f"Warning: Compressed not found: {compressed_path}")
            continue
        
        # Create sequential filenames for easier navigation
        seq_num = len(copied_images) + 1
        # Include true class and predictions in filename for quick reference
        safe_filename = f"{seq_num:04d}_{name_without_ext}"
        
        # Copy to temp folders
        dst_original = temp_uncomp / f"{safe_filename}.jpg"
        dst_compressed = temp_comp / f"{safe_filename}.png"
        
        shutil.copy2(original_path, dst_original)
        shutil.copy2(compressed_path, dst_compressed)
        
        # Handle aggregated fields (may be lists from groupby)
        compressed_pred = row['compressed_pred']
        if isinstance(compressed_pred, list):
            compressed_pred = ', '.join(str(p) for p in compressed_pred)
        
        models_affected = row.get('model', 'unknown')
        if isinstance(models_affected, list):
            models_affected = ', '.join(str(m) for m in models_affected)
        
        copied_images.append({
            'sequence': seq_num,
            'filename': filename,
            'true_class': row['true_class'],
            'original_pred': row['original_pred'],
            'compressed_pred': compressed_pred,
            'original_confidence': row['original_confidence'],
            'compressed_confidence': row['compressed_confidence'],
            'models_affected': models_affected
        })
    
    print(f"Prepared {len(copied_images)} image pairs in temporary folders")
    
    return temp_uncomp, temp_comp, copied_images

def create_session_csv(quality, codec, copied_images):
    """Create a CSV file documenting all images in this labelling session."""
    session_dir = LABELLING_DIR / "sessions"
    session_dir.mkdir(exist_ok=True)
    
    # Create session filename
    session_name = f"session_{codec}_{quality}"
    csv_path = session_dir / f"{session_name}.csv"
    
    # Create DataFrame with image info
    df = pd.DataFrame(copied_images)
    df['quality'] = quality
    df['codec'] = codec
    
    # Reorder columns
    columns = ['sequence', 'filename', 'quality', 'codec', 
               'true_class', 'original_pred', 'compressed_pred',
               'original_confidence', 'compressed_confidence', 'models_affected']
    df = df[columns]
    
    df.to_csv(csv_path, index=False)
    print(f"\nSession CSV created: {csv_path}")
    
    # Open CSV in default application
    subprocess.run(['open', str(csv_path)])
    
    return csv_path, session_name

def run_vpv_with_tag_capture(temp_uncomp, temp_comp, label_output_path):
    
    # Change to labelling directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(LABELLING_DIR)
    
    try:
        with open(label_output_path, 'a') as tag_file:
            # Write header
            tag_file.write(f"# VPV Tags captured from labelling session\n")
            tag_file.write(f"# Folder: temp_uncomp vs temp_comp\n")
            tag_file.flush()
            
            # Use pseudoterminal to capture VPV output without buffering
            master_fd, slave_fd = pty.openpty()
            
            # Launch VPV with both folders
            # 'aw' = auto-window, 'nw' = new window,
            process = subprocess.Popen(
                [str(VPV_PATH), "aw", "temp_comp", "temp_uncomp", "nw", "temp_uncomp"],
                stdout=slave_fd,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            os.close(slave_fd)
            
            with os.fdopen(master_fd) as master:
                for line in master:
                    # Print to terminal for user feedback
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    
                    # Capture TAG lines
                    if line.startswith("TAG"):
                        tag_file.write(line)
                        tag_file.flush()
            
            process.wait()
            
    finally:
        os.chdir(original_cwd)
    
    return label_output_path

def main():
    """Main entry point."""
    
    # Check if misclassifications CSV exists
    if not MISCLASSIFICATIONS_CSV.exists():
        print(f"\nError: Cannot find misclassifications CSV at:")
        print(f"  {MISCLASSIFICATIONS_CSV}")
        print("\nPlease run the miscompression identification script first.")
        sys.exit(1)
    
    # Get user selection (quality is fixed to q6, models are combined)
    codec = get_user_selection()
    quality = QUALITY
    
    print(f"\nSelected configuration:")
    print(f"  Quality: {quality} (fixed)")
    print(f"  Codec:   {codec}")
    print(f"  Models:  combined (both ResNet-18 and ViT-S/16)")
    
    # Load misclassifications for this codec (unique images across models)
    misclassifications = load_misclassifications(quality, codec)
    
    if len(misclassifications) == 0:
        print(f"\nNo misclassifications found for this configuration.")
        print("This could mean the codec performed perfectly at this quality level.")
        sys.exit(0)
    
    # Prepare images in temporary folders
    temp_uncomp, temp_comp, copied_images = prepare_images(misclassifications, quality, codec)
    
    if len(copied_images) == 0:
        print("\nNo images could be prepared. Check that image files exist.")
        sys.exit(1)
    
    # Create session CSV
    csv_path, session_name = create_session_csv(quality, codec, copied_images)
    
    # Create label output file
    labels_dir = LABELLING_DIR / "labels"
    labels_dir.mkdir(exist_ok=True)
    label_file = labels_dir / f"labels_{session_name}.txt"
    
    # Run VPV and capture tags
    run_vpv_with_tag_capture(temp_uncomp, temp_comp, label_file)
    
    # Cleanup message
    print(f"\nSession CSV:  {csv_path}")
    print(f"Labels file:  {label_file}")
    print(f"\nTemporary folders are preserved at:")
    print(f"  {temp_uncomp}")
    print(f"  {temp_comp}")
    print("\nTo clean up temporary folders, run with --cleanup flag")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        print("Cleaning up temporary folders...")
        clear_temp_folders()
        print("Done.")
    else:
        main()
