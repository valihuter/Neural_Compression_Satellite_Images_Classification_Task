#!/usr/bin/env python3
"""
Generate publication-ready figures comparing ViT-S/16 and ResNet-18
on compressed satellite images.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setup
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'

# Load data
df = pd.read_csv(RESULTS_DIR / 'vit_vs_resnet_full_comparison.csv')

# Baselines
VIT_BASELINE = 99.73
RESNET_BASELINE = 97.65

# =============================================================================
# Figure 1: Rate-Accuracy Curves (Main Result)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'JPEG': ('#1f77b4', '#aec7e8'),      # Blue
    'Cheng2020': ('#2ca02c', '#98df8a'),  # Green
    'MS-ILLM': ('#d62728', '#ff9896'),    # Red
    'JPEG-AI': ('#9467bd', '#c5b0d5')     # Purple
}

markers_vit = {'JPEG': 'o', 'Cheng2020': 's', 'MS-ILLM': '^', 'JPEG-AI': 'D'}
markers_resnet = {'JPEG': 'o', 'Cheng2020': 's', 'MS-ILLM': '^', 'JPEG-AI': 'D'}

for codec in df['Codec'].unique():
    codec_df = df[df['Codec'] == codec].sort_values('BPP')
    color_vit, color_resnet = colors.get(codec, ('#333333', '#999999'))
    
    # ViT (solid line)
    ax.plot(codec_df['BPP'], codec_df['ViT_Accuracy'], 
            marker=markers_vit[codec], markersize=8, linewidth=2,
            color=color_vit, label=f'{codec} (ViT)')
    
    # ResNet (dashed line)
    ax.plot(codec_df['BPP'], codec_df['ResNet_Accuracy'], 
            marker=markers_resnet[codec], markersize=8, linewidth=2,
            linestyle='--', color=color_resnet, alpha=0.7, label=f'{codec} (ResNet)')

# Baselines
ax.axhline(y=VIT_BASELINE, color='black', linestyle=':', linewidth=1.5, 
           label=f'ViT Baseline ({VIT_BASELINE}%)')
ax.axhline(y=RESNET_BASELINE, color='gray', linestyle=':', linewidth=1.5,
           label=f'ResNet Baseline ({RESNET_BASELINE}%)')

ax.set_xlabel('Bitrate (BPP)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('ViT-S/16 vs ResNet-18: Rate-Accuracy on Compressed Satellite Images')
ax.set_xlim(0, 1.6)
ax.set_ylim(25, 102)
ax.legend(loc='lower right', ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_rate_accuracy.pdf')
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_rate_accuracy.png')
print("Saved: vit_vs_resnet_rate_accuracy.pdf/png")
plt.close()

# =============================================================================
# Figure 2: Accuracy Drop Comparison (Bar Chart)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

codecs = ['JPEG', 'Cheng2020', 'MS-ILLM', 'JPEG-AI']

for idx, codec in enumerate(codecs):
    ax = axes[idx]
    codec_df = df[df['Codec'] == codec].sort_values('BPP')
    
    x = np.arange(len(codec_df))
    width = 0.35
    
    # Short quality names
    quality_names = [q.split('_')[-1] for q in codec_df['Quality']]
    
    bars1 = ax.bar(x - width/2, codec_df['ViT_Drop'], width, 
                   label='ViT-S/16', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, codec_df['ResNet_Drop'], width,
                   label='ResNet-18', color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Quality Level')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title(f'{codec}')
    ax.set_xticks(x)
    ax.set_xticklabels(quality_names)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(codec_df['ViT_Drop'].max(), codec_df['ResNet_Drop'].max()) * 1.1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.suptitle('Accuracy Drop by Compression Level: ViT-S/16 vs ResNet-18', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_accuracy_drop.pdf')
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_accuracy_drop.png')
print("Saved: vit_vs_resnet_accuracy_drop.pdf/png")
plt.close()

# =============================================================================
# Figure 3: Robustness Gap (ViT - ResNet Difference)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for codec in df['Codec'].unique():
    codec_df = df[df['Codec'] == codec].sort_values('BPP')
    color = colors.get(codec, ('#333333', '#999999'))[0]
    
    ax.plot(codec_df['BPP'], codec_df['ViT_vs_ResNet'], 
            marker='o', markersize=8, linewidth=2,
            color=color, label=codec)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=2.08, color='green', linestyle='--', linewidth=1.5, 
           label='Baseline Advantage (+2.08%)')

ax.fill_between([0, 1.6], 0, 10, alpha=0.1, color='green', label='ViT better')
ax.fill_between([0, 1.6], 0, -15, alpha=0.1, color='red', label='ResNet better')

ax.set_xlabel('Bitrate (BPP)')
ax.set_ylabel('ViT - ResNet Accuracy Difference (%)')
ax.set_title('Compression Robustness Gap: ViT-S/16 vs ResNet-18')
ax.set_xlim(0, 1.6)
ax.set_ylim(-12, 5)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_robustness_gap.pdf')
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_robustness_gap.png')
print("Saved: vit_vs_resnet_robustness_gap.pdf/png")
plt.close()

# =============================================================================
# Figure 4: Summary Bar Chart (Best Quality per Codec)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get highest quality for each codec
summary_data = []
for codec in ['JPEG', 'Cheng2020', 'MS-ILLM', 'JPEG-AI']:
    codec_df = df[df['Codec'] == codec]
    best_row = codec_df.loc[codec_df['BPP'].idxmax()]
    summary_data.append({
        'Codec': codec,
        'Quality': best_row['Quality'].split('_')[-1],
        'BPP': best_row['BPP'],
        'ViT': best_row['ViT_Accuracy'],
        'ResNet': best_row['ResNet_Accuracy']
    })

summary_df = pd.DataFrame(summary_data)

x = np.arange(len(summary_df))
width = 0.35

bars1 = ax.bar(x - width/2, summary_df['ViT'], width, 
               label='ViT-S/16', color='#1f77b4')
bars2 = ax.bar(x + width/2, summary_df['ResNet'], width,
               label='ResNet-18', color='#ff7f0e')

# Baselines
ax.axhline(y=VIT_BASELINE, color='#1f77b4', linestyle='--', alpha=0.5)
ax.axhline(y=RESNET_BASELINE, color='#ff7f0e', linestyle='--', alpha=0.5)

ax.set_xlabel('Codec (Highest Quality)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Best-Case Accuracy by Codec: ViT-S/16 vs ResNet-18')
ax.set_xticks(x)
ax.set_xticklabels([f"{row['Codec']}\n({row['Quality']}, {row['BPP']:.2f} BPP)" 
                    for _, row in summary_df.iterrows()])
ax.set_ylim(75, 102)
ax.legend(loc='lower right')

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_best_quality.pdf')
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_best_quality.png')
print("Saved: vit_vs_resnet_best_quality.pdf/png")
plt.close()

# =============================================================================
# Figure 5: Heatmap of Accuracy Differences
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

# Pivot data for heatmap
pivot_data = df.pivot(index='Codec', columns='Quality', values='ViT_vs_ResNet')

# Reorder columns by BPP
quality_order = df.sort_values('BPP')['Quality'].unique()
pivot_data = pivot_data[[q for q in quality_order if q in pivot_data.columns]]

# Create heatmap
im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', 
               vmin=-10, vmax=5)

ax.set_xticks(np.arange(len(pivot_data.columns)))
ax.set_yticks(np.arange(len(pivot_data.index)))
ax.set_xticklabels([q.split('_')[-1] for q in pivot_data.columns], rotation=45, ha='right')
ax.set_yticklabels(pivot_data.index)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('ViT - ResNet Accuracy (%)', rotation=-90, va='bottom')

# Add text annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        val = pivot_data.iloc[i, j]
        if not np.isnan(val):
            text = ax.text(j, i, f'{val:+.1f}',
                          ha='center', va='center', 
                          color='white' if abs(val) > 5 else 'black',
                          fontsize=9, fontweight='bold')

ax.set_title('ViT vs ResNet Accuracy Difference by Codec and Quality\n(Green = ViT better, Red = ResNet better)')
ax.set_xlabel('Quality Level')
ax.set_ylabel('Codec')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_heatmap.pdf')
plt.savefig(RESULTS_DIR / 'vit_vs_resnet_heatmap.png')
print("Saved: vit_vs_resnet_heatmap.pdf/png")
plt.close()

print("\nAll figures generated successfully!")
print(f"Output directory: {RESULTS_DIR}")
