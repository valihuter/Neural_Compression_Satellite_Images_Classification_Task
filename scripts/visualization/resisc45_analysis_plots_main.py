#!/usr/bin/env python3
"""
Consolidated plot generation script for both EuroSAT and RESISC45.
Reads all data from master_results.json for easy reproducibility.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR

# Load master results
def load_master_results():
    with open(RESULTS_DIR / "master_results.json") as f:
        return json.load(f)

def load_bpp_data():
    with open(RESULTS_DIR / "measured_bpp_all_codecs.json") as f:
        return json.load(f)

# Color scheme
CODEC_COLORS = {
    'jpeg': '#e41a1c',       # Red
    'jpeg2000': '#377eb8',   # Blue
    'cheng2020': '#4daf4a',  # Green
    'msillm': '#984ea3',     # Purple
    'jpegai': '#ff7f00',     # Orange
    'jpeg_ai': '#ff7f00'     # Orange (alias)
}

CODEC_LABELS = {
    'jpeg': 'JPEG',
    'jpeg2000': 'JPEG2000',
    'cheng2020': 'Cheng2020',
    'msillm': 'MS-ILLM',
    'jpegai': 'JPEG-AI',
    'jpeg_ai': 'JPEG-AI'
}

EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

RESISC45_CLASSES = [
    'beach', 'circular_farmland', 'dense_residential', 'forest', 'freeway',
    'industrial_area', 'lake', 'meadow', 'medium_residential', 
    'rectangular_farmland', 'river'
]

def get_eurosat_bpp(bpp_data, codec):
    """Get BPP values for EuroSAT."""
    eurosat = bpp_data.get("eurosat", {})
    codec_key = codec.replace("jpeg_ai", "jpegai")
    return eurosat.get(codec_key, {})

def get_resisc45_bpp(bpp_data, codec):
    """Get BPP values for RESISC45."""
    resisc = bpp_data.get("resisc45", {})
    codec_key = codec.replace("jpeg_ai", "jpegai")
    return resisc.get(codec_key, {})

def extract_eurosat_accuracy(master, model, codec):
    """Extract accuracy data for EuroSAT."""
    data = master.get("eurosat", {}).get(model, {}).get(codec, {})
    accuracies = {}
    for q, vals in data.items():
        if isinstance(vals, dict) and "accuracy" in vals:
            accuracies[q] = vals["accuracy"]
    return accuracies

def extract_resisc45_accuracy(master, model, codec):
    """Extract accuracy data for RESISC45."""
    data = master.get("resisc45", {}).get(model, {}).get(codec, {})
    accuracies = {}
    for q, vals in data.items():
        if isinstance(vals, dict) and "overall_accuracy" in vals:
            accuracies[q] = vals["overall_accuracy"]
    return accuracies

# ============================================================
# EUROSAT PLOTS
# ============================================================

def plot_eurosat_rate_accuracy(master, bpp_data):
    """Rate-accuracy plot for EuroSAT."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for codec in ['jpeg', 'jpeg2000', 'cheng2020', 'msillm', 'jpeg_ai']:
        bpps_dict = get_eurosat_bpp(bpp_data, codec)
        accs_dict = extract_eurosat_accuracy(master, 'vit_s16', codec)
        
        bpps = []
        accs = []
        for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
            q_key = q
            # Handle different naming conventions
            if codec == 'jpeg':
                q_key = f'jpeg_{q}'
            elif codec == 'jpeg2000':
                q_key = f'jpeg2000_{q}'
            elif codec == 'cheng2020':
                q_key = f'cheng2020-attn_{q}'
            elif codec == 'msillm':
                q_key = f'msillm_{q}'
            elif codec == 'jpeg_ai':
                q_key = f'jpeg_ai_{q}'
            
            if q_key in accs_dict and q in bpps_dict:
                bpps.append(bpps_dict[q])
                accs.append(accs_dict[q_key])
        
        if bpps and accs:
            ax.plot(bpps, accs, 'o-', color=CODEC_COLORS.get(codec, 'gray'), 
                   label=CODEC_LABELS.get(codec, codec), linewidth=2, markersize=8)
    
    ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('EuroSAT: Rate-Accuracy Comparison (ViT-S/16)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'eurosat_rate_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'eurosat_rate_accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("eurosat_rate_accuracy_comparison.png/pdf")

def plot_resisc45_rate_accuracy(master, bpp_data):
    """Rate-accuracy plot for RESISC45."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    resisc_bpp = bpp_data.get("resisc45", {})
    
    for idx, (model, model_name) in enumerate([('resnet18', 'ResNet-18'), ('vit_s16', 'ViT-S/16')]):
        ax = axes[idx]
        
        for codec in ['jpeg', 'jpeg2000', 'cheng2020', 'msillm', 'jpegai']:
            codec_bpp = resisc_bpp.get(codec, {})
            accs_dict = extract_resisc45_accuracy(master, model, codec)
            
            bpps = []
            accs = []
            for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
                if q in accs_dict and q in codec_bpp:
                    bpp_val = codec_bpp[q]
                    if bpp_val is not None:
                        bpps.append(bpp_val)
                        accs.append(accs_dict[q])
            
            if bpps and accs:
                ax.plot(bpps, accs, 'o-', color=CODEC_COLORS.get(codec, 'gray'), 
                       label=CODEC_LABELS.get(codec, codec), linewidth=2, markersize=8)
        
        ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
        ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
        ax.set_title(f'{model_name} - RESISC45', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(85, 100)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'resisc45_rate_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'resisc45_rate_accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("resisc45_rate_accuracy_comparison.png/pdf")

def plot_resisc45_vit_vs_resnet(master, bpp_data):
    """ViT vs ResNet comparison for RESISC45."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    resisc_bpp = bpp_data.get("resisc45", {})
    
    for codec in ['jpeg', 'jpeg2000', 'cheng2020', 'msillm', 'jpegai']:
        codec_bpp = resisc_bpp.get(codec, {})
        
        # ResNet
        resnet_accs = extract_resisc45_accuracy(master, 'resnet18', codec)
        bpps_r, accs_r = [], []
        for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
            if q in resnet_accs and q in codec_bpp and codec_bpp[q] is not None:
                bpps_r.append(codec_bpp[q])
                accs_r.append(resnet_accs[q])
        
        # ViT
        vit_accs = extract_resisc45_accuracy(master, 'vit_s16', codec)
        bpps_v, accs_v = [], []
        for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
            if q in vit_accs and q in codec_bpp and codec_bpp[q] is not None:
                bpps_v.append(codec_bpp[q])
                accs_v.append(vit_accs[q])
        
        if bpps_r and accs_r:
            ax.plot(bpps_r, accs_r, '--', color=CODEC_COLORS.get(codec, 'gray'), linewidth=2, alpha=0.7)
        if bpps_v and accs_v:
            ax.plot(bpps_v, accs_v, 'o-', color=CODEC_COLORS.get(codec, 'gray'), 
                   label=CODEC_LABELS.get(codec, codec), linewidth=2, markersize=8)
    
    # Add legend entries for line styles
    ax.plot([], [], 'k-', linewidth=2, label='ViT-S/16 (solid)')
    ax.plot([], [], 'k--', linewidth=2, alpha=0.7, label='ResNet-18 (dashed)')
    
    ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('RESISC45: ViT-S/16 vs ResNet-18', fontsize=14)
    ax.legend(loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(85, 100)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'resisc45_vit_vs_resnet_rate_accuracy.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'resisc45_vit_vs_resnet_rate_accuracy.pdf', bbox_inches='tight')
    plt.close()
    print("resisc45_vit_vs_resnet_rate_accuracy.png/pdf")

def plot_resisc45_jpegai_detailed(master, bpp_data):
    """Detailed JPEG-AI plot for RESISC45."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bpps = [0.12, 0.25, 0.50, 0.75, 1.00, 1.50]
    
    jpegai_resnet = master["resisc45"]["resnet18"].get("jpegai", {})
    jpegai_vit = master["resisc45"]["vit_s16"].get("jpegai", {})
    
    resnet_accs = [jpegai_resnet.get(f'q{i}', {}).get('overall_accuracy', 0) for i in range(1, 7)]
    vit_accs = [jpegai_vit.get(f'q{i}', {}).get('overall_accuracy', 0) for i in range(1, 7)]
    
    ax.plot(bpps, resnet_accs, 'o--', color='#e41a1c', label='ResNet-18', linewidth=2, markersize=10)
    ax.plot(bpps, vit_accs, 's-', color='#377eb8', label='ViT-S/16', linewidth=2, markersize=10)
    
    # Add annotations
    for i, (bpp, r_acc, v_acc) in enumerate(zip(bpps, resnet_accs, vit_accs)):
        ax.annotate(f'{r_acc:.1f}%', (bpp, r_acc), textcoords="offset points", 
                   xytext=(0, -15), ha='center', fontsize=9, color='#e41a1c')
        ax.annotate(f'{v_acc:.1f}%', (bpp, v_acc), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color='#377eb8')
    
    ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('JPEG-AI Performance on RESISC45', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(88, 100)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'resisc45_jpegai_detailed.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'resisc45_jpegai_detailed.pdf', bbox_inches='tight')
    plt.close()
    print("resisc45_jpegai_detailed.png/pdf")

def plot_resisc45_class_heatmap(master):
    """Class accuracy heatmap for JPEG-AI on RESISC45."""
    jpegai_vit = master["resisc45"]["vit_s16"].get("jpegai", {})
    
    matrix = []
    for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']:
        if q in jpegai_vit:
            row = [jpegai_vit[q].get('per_class_accuracy', {}).get(cls, 0) for cls in RESISC45_CLASSES]
            matrix.append(row)
    
    if not matrix:
        print(" No data for RESISC45 heatmap")
        return
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=70, vmax=100)
    
    ax.set_xticks(range(len(RESISC45_CLASSES)))
    ax.set_xticklabels([c.replace('_', '\n') for c in RESISC45_CLASSES], rotation=45, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels(['q1 (0.12)', 'q2 (0.25)', 'q3 (0.50)', 'q4 (0.75)', 'q5 (1.00)', 'q6 (1.50)'])
    
    for i in range(len(matrix)):
        for j in range(len(RESISC45_CLASSES)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}', ha='center', va='center', 
                          color='white' if matrix[i, j] < 85 else 'black', fontsize=9)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Quality Level (BPP)', fontsize=12)
    ax.set_title('JPEG-AI Class Accuracy Heatmap - RESISC45 (ViT-S/16)', fontsize=14)
    
    plt.colorbar(im, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'resisc45_jpegai_class_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'resisc45_jpegai_class_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("resisc45_jpegai_class_heatmap.png/pdf")

def plot_cross_dataset_comparison(master, bpp_data):
    """Compare JPEG-AI performance across EuroSAT and RESISC45."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # RESISC45 JPEG-AI
    resisc_bpps = [0.12, 0.25, 0.50, 0.75, 1.00, 1.50]
    resisc_vit = master["resisc45"]["vit_s16"].get("jpegai", {})
    resisc_accs = [resisc_vit.get(f'q{i}', {}).get('overall_accuracy', 0) for i in range(1, 7)]
    
    # EuroSAT JPEG-AI (from vit_s16)
    eurosat_bpps = [0.13, 0.25, 0.50, 0.75, 1.00, 1.50]
    eurosat_vit = master["eurosat"]["vit_s16"].get("jpeg_ai", {})
    eurosat_accs = []
    for i in range(1, 7):
        q_key = f'jpeg_ai_q{i}'
        if q_key in eurosat_vit:
            eurosat_accs.append(eurosat_vit[q_key].get('accuracy', 0))
        else:
            eurosat_accs.append(0)
    
    ax.plot(resisc_bpps, resisc_accs, 'o-', color='#377eb8', label='RESISC45 (256×256)', linewidth=2, markersize=8)
    ax.plot(eurosat_bpps, eurosat_accs, 's--', color='#e41a1c', label='EuroSAT (64×64)', linewidth=2, markersize=8)
    
    ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('JPEG-AI: Cross-Dataset Comparison (ViT-S/16)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(70, 100)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cross_dataset_jpegai_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / 'cross_dataset_jpegai_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("cross_dataset_jpegai_comparison.png/pdf")

def main():
    print("Generating Plots from Master Results")
    
    master = load_master_results()
    bpp_data = load_bpp_data()
    
    print(f"
EuroSAT Plots")
    try:
        plot_eurosat_rate_accuracy(master, bpp_data)
    except Exception as e:
        print(f" EuroSAT rate-accuracy failed: {e}")
    
    print(f"
RESISC45 Plots")
    plot_resisc45_rate_accuracy(master, bpp_data)
    plot_resisc45_vit_vs_resnet(master, bpp_data)
    plot_resisc45_jpegai_detailed(master, bpp_data)
    plot_resisc45_class_heatmap(master)
    
    print(f"
Cross-Dataset Plots")
    plot_cross_dataset_comparison(master, bpp_data)
    
    print("\nAll plots generated from master_results.json!")

if __name__ == "__main__":
    main()
