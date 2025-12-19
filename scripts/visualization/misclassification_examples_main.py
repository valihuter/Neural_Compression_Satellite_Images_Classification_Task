#!/usr/bin/env python3
"""
Generate misclassification example figures for thesis.
Shows original vs compressed images that were misclassified.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # Go up to repo root
RESULTS_DIR = BASE_DIR / "results"
THESIS_IMG_DIR = BASE_DIR / "docs" / "thesis_fhkufstein" / "img"

EUROSAT_UNCOMP = BASE_DIR / "data" / "eurosat" / "raw"
EUROSAT_COMP = BASE_DIR / "data" / "eurosat" / "comp"

# RESISC45 paths
RESISC45_UNCOMP = BASE_DIR / "data" / "resisc45" / "raw" / "NWPU-RESISC45"
RESISC45_COMP = BASE_DIR / "data" / "resisc45" / "comp"

EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def create_eurosat_misclassification_examples():
    """Create figure showing EuroSAT misclassification examples."""
    print("Creating EuroSAT misclassification examples...")
    
    # Known problematic patterns from analysis:
    # Forest → SeaLake (most common, 19% of all misclassifications)
    # Residential → HerbaceousVegetation
    # PermanentCrop → HerbaceousVegetation
    
    examples = [
        ('Forest', 'SeaLake', 'forest_sealake'),
        ('Residential', 'HerbaceousVegetation', 'residential_herbveg'),
        ('PermanentCrop', 'HerbaceousVegetation', 'permcrop_herbveg'),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    
    for row, (true_class, confused_class, pattern_name) in enumerate(examples):
        # Find images that exist in both original and compressed folders
        orig_dir = EUROSAT_UNCOMP / true_class
        comp_dir = EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q1' / true_class

        if orig_dir.exists() and comp_dir.exists():
            # Get filenames from both folders
            orig_images = {p.stem for p in orig_dir.glob("*.jpg")}
            comp_images = {p.stem for p in comp_dir.glob("*.jpg")}

            # Find intersection
            common_images = sorted(orig_images & comp_images)

            if len(common_images) > 10:
                # Pick an image in the middle
                idx = len(common_images) // 2
                img_name = common_images[idx]
                orig_path = orig_dir / f"{img_name}.jpg"
                
                orig_img = Image.open(orig_path)

                # Column 0: Original
                axes[row, 0].imshow(orig_img)
                if row == 0:
                    axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
                axes[row, 0].set_ylabel(f"True: {true_class}", fontsize=10)
                axes[row, 0].axis('off')
                
                # Column 1: JPEG-AI q1
                # Try different possible paths for compressed images
                possible_comp_paths = [
                    EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q1' / true_class / f"{img_name}.jpg",
                    EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q1' / true_class / f"{img_name}.png",
                    EUROSAT_COMP / 'jpeg_ai' / 'q1' / true_class / f"{img_name}.jpg",
                    EUROSAT_COMP / 'jpegai' / 'q1' / true_class / f"{img_name}.jpg",
                ]
                
                comp_found = False
                for comp_path in possible_comp_paths:
                    if comp_path.exists():
                        comp_img = Image.open(comp_path)
                        axes[row, 1].imshow(comp_img)
                        comp_found = True
                        break

                if not comp_found:
                    axes[row, 1].text(0.5, 0.5, 'Not found', ha='center', va='center')
                
                if row == 0:
                    axes[row, 1].set_title('JPEG-AI q1\n(0.13 BPP)', fontsize=11, fontweight='bold')
                axes[row, 1].axis('off')
                
                # Column 2: JPEG-AI q6
                possible_q6_paths = [
                    EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q6' / true_class / f"{img_name}.jpg",
                    EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q6' / true_class / f"{img_name}.png",
                    EUROSAT_COMP / 'jpeg_ai' / 'q6' / true_class / f"{img_name}.jpg",
                    EUROSAT_COMP / 'jpegai' / 'q6' / true_class / f"{img_name}.jpg",
                ]
                
                q6_found = False
                for q6_path in possible_q6_paths:
                    if q6_path.exists():
                        q6_img = Image.open(q6_path)
                        axes[row, 2].imshow(q6_img)
                        q6_found = True
                        break
                
                if not q6_found:
                    axes[row, 2].text(0.5, 0.5, 'Not found', ha='center', va='center')
                
                if row == 0:
                    axes[row, 2].set_title('JPEG-AI q6\n(1.50 BPP)', fontsize=11, fontweight='bold')
                axes[row, 2].axis('off')
                
                # Column 3: Example of confused class
                conf_dir = EUROSAT_UNCOMP / confused_class
                if conf_dir.exists():
                    conf_images = sorted(conf_dir.glob("*.jpg"))
                    if conf_images:
                        conf_img = Image.open(conf_images[0])
                        axes[row, 3].imshow(conf_img)
                
                if row == 0:
                    axes[row, 3].set_title(f'Confused Class:\n{confused_class}', fontsize=11, fontweight='bold')
                else:
                    axes[row, 3].set_title(f'{confused_class}', fontsize=10)
                axes[row, 3].axis('off')
    
    plt.suptitle('EuroSAT: Compression-Induced Misclassification Patterns', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(THESIS_IMG_DIR / 'eurosat_misclassification_examples.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(RESULTS_DIR / 'eurosat_misclassification_examples.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved eurosat_misclassification_examples.pdf")

def create_resisc45_misclassification_detailed():
    """Create detailed misclassification examples for RESISC45."""
    print("Creating RESISC45 misclassification examples...")
    
    # Based on analysis, most errors at q1 are in:
    # - circular_farmland (90% accuracy → 10% error rate)
    # - dense_residential (93% accuracy → 7% error rate)
    
    examples = [
        ('circular_farmland', 'rectangular_farmland'),
        ('dense_residential', 'medium_residential'),
        ('meadow', 'forest'),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    
    for row, (true_class, confused_class) in enumerate(examples):
        # Find images that exist in both original and compressed folders
        orig_dir = RESISC45_UNCOMP / true_class
        comp_dir = RESISC45_COMP / 'RESISC45_SUBSET11_JPEGAI' / 'q1' / true_class

        if orig_dir.exists() and comp_dir.exists():
            # Get filenames from both folders
            orig_images = {p.stem for p in orig_dir.glob("*.jpg")}
            comp_images = {p.stem for p in comp_dir.glob("*.png")}  # RESISC45 compressed are PNG

            # Find intersection
            common_images = sorted(orig_images & comp_images)

            if len(common_images) > 10:
                idx = len(common_images) // 2
                img_name = common_images[idx]
                orig_path = orig_dir / f"{img_name}.jpg"
                
                orig_img = Image.open(orig_path)
                
                # Column 0: Original
                axes[row, 0].imshow(orig_img)
                if row == 0:
                    axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
                axes[row, 0].set_ylabel(f"True: {true_class.replace('_', ' ')}", fontsize=9)
                axes[row, 0].axis('off')
                
                # Column 1: JPEG-AI q1
                q1_paths = [
                    RESISC45_COMP / 'RESISC45_SUBSET11_JPEGAI' / 'q1' / true_class / f"{img_name}.jpg",
                    RESISC45_COMP / 'RESISC45_SUBSET11_JPEGAI' / 'q1' / true_class / f"{img_name}.png",
                ]
                q1_found = False
                for q1_path in q1_paths:
                    if q1_path.exists():
                        comp_img = Image.open(q1_path)
                        axes[row, 1].imshow(comp_img)
                        q1_found = True
                        break
                if not q1_found:
                    axes[row, 1].text(0.5, 0.5, 'Not found', ha='center', va='center')

                if row == 0:
                    axes[row, 1].set_title('JPEG-AI q1\n(0.12 BPP)', fontsize=11, fontweight='bold')
                axes[row, 1].axis('off')

                # Column 2: JPEG-AI q6
                q6_paths = [
                    RESISC45_COMP / 'RESISC45_SUBSET11_JPEGAI' / 'q6' / true_class / f"{img_name}.jpg",
                    RESISC45_COMP / 'RESISC45_SUBSET11_JPEGAI' / 'q6' / true_class / f"{img_name}.png",
                ]
                q6_found = False
                for q6_path in q6_paths:
                    if q6_path.exists():
                        q6_img = Image.open(q6_path)
                        axes[row, 2].imshow(q6_img)
                        q6_found = True
                        break
                if not q6_found:
                    axes[row, 2].text(0.5, 0.5, 'Not found', ha='center', va='center')
                
                if row == 0:
                    axes[row, 2].set_title('JPEG-AI q6\n(1.50 BPP)', fontsize=11, fontweight='bold')
                axes[row, 2].axis('off')
                
                # Column 3: Confused class example
                conf_dir = RESISC45_UNCOMP / confused_class
                if conf_dir.exists():
                    conf_images = sorted(conf_dir.glob("*.jpg"))
                    if conf_images:
                        conf_img = Image.open(conf_images[0])
                        axes[row, 3].imshow(conf_img)
                
                if row == 0:
                    axes[row, 3].set_title(f'Confused Class', fontsize=11, fontweight='bold')
                axes[row, 3].set_xlabel(confused_class.replace('_', ' '), fontsize=9)
                axes[row, 3].axis('off')
    
    plt.suptitle('RESISC45: Compression-Induced Misclassification Patterns', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(THESIS_IMG_DIR / 'resisc45_misclassification_detailed.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(RESULTS_DIR / 'resisc45_misclassification_detailed.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved resisc45_misclassification_detailed.pdf")

def create_side_by_side_codec_comparison():
    """Create side-by-side comparison of all codecs on same image."""
    print("Creating codec comparison figure...")
    
    # Find a Forest image that exists in all codec folders
    forest_dir = EUROSAT_UNCOMP / 'Forest'
    if not forest_dir.exists():
        print("Forest directory not found")
        return

    # Define codec directories to check
    codec_check_dirs = [
        EUROSAT_COMP / 'jpeg_ai' / 'jpeg_ai_q3' / 'Forest',
        EUROSAT_COMP / 'cheng2020-attn' / 'cheng2020-attn_q3' / 'Forest',
        EUROSAT_COMP / 'msillm' / 'msillm_q3' / 'Forest',
    ]

    # Get images available in all neural codec folders (JPEG2000 has all images, skip it)
    all_codec_images = []
    for codec_dir in codec_check_dirs:
        if codec_dir.exists():
            images = {p.stem for p in codec_dir.glob("*.jpg")} | {p.stem for p in codec_dir.glob("*.png")}
            all_codec_images.append(images)

    if not all_codec_images:
        print("No codec directories found")
        return

    # Find intersection of images available in all codecs
    common_images = sorted(set.intersection(*all_codec_images))

    if not common_images:
        print("No common Forest images found across all codecs")
        return

    # Pick an image
    idx = len(common_images) // 2
    img_name = common_images[idx]
    orig_path = forest_dir / f"{img_name}.jpg"
    
    # Map codec names to their folder structures
    # Note: JPEG2000 excluded due to color space encoding issues in .jp2 files
    codec_paths = [
        ('jpeg_compressed', 'jpeg_q50', 'JPEG'),
        ('cheng2020-attn', 'cheng2020-attn_q3', 'Cheng2020'),
        ('msillm', 'msillm_q3', 'MS-ILLM'),
        ('jpeg_ai', 'jpeg_ai_q3', 'JPEG-AI'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Original
    orig_img = Image.open(orig_path)
    axes[0].imshow(orig_img)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Compressed versions
    for i, (codec_folder, quality_folder, label) in enumerate(codec_paths, 1):
        possible_paths = [
            EUROSAT_COMP / codec_folder / quality_folder / 'Forest' / f"{img_name}.jpg",
            EUROSAT_COMP / codec_folder / quality_folder / 'Forest' / f"{img_name}.png",
            EUROSAT_COMP / codec_folder / quality_folder / 'Forest' / f"{img_name}.jp2",  # JPEG2000 format
        ]

        found = False
        for path in possible_paths:
            if path.exists():
                comp_img = Image.open(path)
                # JPEG2000 files might be in YCbCr mode, convert to RGB
                if comp_img.mode == 'YCbCr':
                    comp_img = comp_img.convert('RGB')
                axes[i].imshow(comp_img)
                found = True
                break

        if not found:
            axes[i].text(0.5, 0.5, 'Not found', ha='center', va='center')
        
        axes[i].set_title(f'{label} (q3)', fontsize=11, fontweight='bold')
        axes[i].axis('off')

    # Hide the last unused subplot (we have 5 images, not 6)
    axes[5].axis('off')

    plt.suptitle('EuroSAT Forest: Neural Codec Comparison at Quality Level q3', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(THESIS_IMG_DIR / 'codec_comparison_forest_all.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(RESULTS_DIR / 'codec_comparison_forest_all.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved codec_comparison_forest_all.pdf")

def main():
    print("Generating Misclassification Example Figures")
    
    create_eurosat_misclassification_examples()
    create_resisc45_misclassification_detailed()
    create_side_by_side_codec_comparison()
    
    print("\nAll misclassification figures complete!")

if __name__ == "__main__":
    main()
