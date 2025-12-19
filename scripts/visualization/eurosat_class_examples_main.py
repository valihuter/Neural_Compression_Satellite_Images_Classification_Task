#!/usr/bin/env python3
"""
Generate a grid of EuroSAT example images for the thesis.
Creates a 2x5 grid showing one example per land cover class.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

DATA_ROOT = Path(__file__).parent.parent / 'data' / 'eurosat' / 'raw'
OUTPUT_DIR = Path(__file__).parent.parent / 'docs' / 'thesis_fhkufstein' / 'img'

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Short labels for display
CLASS_LABELS = [
    'Annual Crop', 'Forest', 'Herbaceous\nVegetation', 'Highway', 'Industrial',
    'Pasture', 'Permanent\nCrop', 'Residential', 'River', 'Sea/Lake'
]

def generate_class_grid():
    """Generate a 2x5 grid of EuroSAT class examples."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
    axes = axes.flatten()
    
    for idx, (cls_name, label) in enumerate(zip(CLASSES, CLASS_LABELS)):
        # Get first image of each class
        img_path = DATA_ROOT / cls_name / f'{cls_name}_1.jpg'
        
        if img_path.exists():
            img = Image.open(img_path)
            axes[idx].imshow(np.array(img))
        else:
            # Fallback: try any image
            cls_dir = DATA_ROOT / cls_name
            if cls_dir.exists():
                img_path = next(cls_dir.glob('*.jpg'), None)
                if img_path:
                    img = Image.open(img_path)
                    axes[idx].imshow(np.array(img))
        
        axes[idx].set_title(label, fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / 'eurosat_class_examples.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'eurosat_class_examples.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'eurosat_class_examples.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'eurosat_class_examples.png'}")
    plt.close()

def generate_compression_comparison():
    """Generate a comparison showing original vs compressed for Forest class."""
    # Forest_749 is one of the universally misclassified images
    img_name = 'Forest_749'
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    
    # Original
    orig_path = DATA_ROOT / 'Forest' / f'{img_name}.jpg'
    if orig_path.exists():
        img = Image.open(orig_path)
        axes[0].imshow(np.array(img))
        axes[0].set_title('Original\n(Forest)', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Cheng2020 q1 (if exists)
    cheng_path = Path(__file__).parent.parent / 'data' / 'compressed_neural' / 'cheng2020-attn' / 'cheng2020-attn_q1' / 'Forest' / f'{img_name}.png'
    if cheng_path.exists():
        img = Image.open(cheng_path)
        axes[1].imshow(np.array(img))
        axes[1].set_title('Cheng2020 q1\n(0.13 BPP)', fontsize=10, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Not found', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].axis('off')
    
    # MS-ILLM q1 (if exists)
    msillm_path = Path(__file__).parent.parent / 'data' / 'compressed_neural' / 'msillm' / 'msillm_q1' / 'Forest' / f'{img_name}.png'
    if msillm_path.exists():
        img = Image.open(msillm_path)
        axes[2].imshow(np.array(img))
        axes[2].set_title('MS-ILLM q1\n(0.035 BPP)', fontsize=10, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Not found', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].axis('off')
    
    # JPEG-AI q1 (if exists)
    jpegai_path = Path(__file__).parent.parent / 'data' / 'compressed_neural' / 'jpeg_ai' / 'jpeg_ai_q1' / 'Forest' / f'{img_name}.png'
    if jpegai_path.exists():
        img = Image.open(jpegai_path)
        axes[3].imshow(np.array(img))
        axes[3].set_title('JPEG-AI q1\n(0.13 BPP)', fontsize=10, fontweight='bold')
    else:
        axes[3].text(0.5, 0.5, 'Not found', ha='center', va='center', transform=axes[3].transAxes)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'compression_comparison_forest.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'compression_comparison_forest.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'compression_comparison_forest.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'compression_comparison_forest.png'}")
    plt.close()

def generate_miscompression_examples():
    """
    Generate visual comparison of universally misclassified images.
    Shows Original vs. compressed versions for the 5 images misclassified across all configurations.
    """
    # The 5 universally misclassified images
    misclassified_images = [
        ('Forest', 'Forest_749'),
        ('PermanentCrop', 'PermanentCrop_1584'),
        ('PermanentCrop', 'PermanentCrop_34'),
        ('PermanentCrop', 'PermanentCrop_661'),
        ('River', 'River_1327'),
    ]
    
    # Create figure: 5 rows (images) x 4 columns (orig + 3 codecs)
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    
    codecs = [
        ('Original', None, None),
        ('Cheng2020 q6', 'cheng2020-attn', 'cheng2020-attn_q6'),
        ('MS-ILLM q6', 'msillm', 'msillm_q6'),
        ('JPEG-AI q6', 'jpeg_ai', 'jpeg_ai_q6'),
    ]
    
    # Column headers
    for col, (codec_name, _, _) in enumerate(codecs):
        axes[0, col].set_title(codec_name, fontsize=11, fontweight='bold', pad=10)
    
    for row, (cls_name, img_name) in enumerate(misclassified_images):
        for col, (codec_name, codec_folder, quality_folder) in enumerate(codecs):
            ax = axes[row, col]
            
            if codec_folder is None:
                # Original image
                img_path = DATA_ROOT / cls_name / f'{img_name}.jpg'
            else:
                # Compressed image
                img_path = Path(__file__).parent.parent / 'data' / 'compressed_neural' / codec_folder / quality_folder / cls_name / f'{img_name}.png'
            
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(np.array(img))
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, color='gray')
            
            ax.axis('off')
            
            # Row labels on the left
            if col == 0:
                # Add class label on the left side
                ax.text(-0.15, 0.5, f'{cls_name}', transform=ax.transAxes,
                       fontsize=9, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    plt.suptitle('Universally Misclassified Images: Original vs. Neural Compression (q6)', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    
    plt.savefig(OUTPUT_DIR / 'miscompression_examples.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'miscompression_examples.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'miscompression_examples.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'miscompression_examples.png'}")
    plt.close()

if __name__ == '__main__':
    print("Generating EuroSAT class examples grid...")
    generate_class_grid()
    
    print("\nGenerating compression comparison...")
    generate_compression_comparison()
    
    print("\nGenerating miscompression examples...")
    generate_miscompression_examples()
    
    print("\nDone!")
    print("\nGenerating compression comparison...")
    generate_compression_comparison()
    
    print("\nDone!")
