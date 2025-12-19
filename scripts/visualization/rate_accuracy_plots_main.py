"""
Comprehensive Codec Comparison for Satellite Image Classification
Generates comparison tables and visualizations for the thesis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load all results
def load_results():
    with open('results/json/master_results.json', 'r') as f:
        return json.load(f)

def extract_data(results):
    """Extract BPP and accuracy data for each codec from master_results.json"""
    data = {}
    # EuroSAT, ViT als Standard
    eurosat = results.get('eurosat', {}).get('vit', {})
    for codec in ['jpegai', 'msillm', 'cheng2020', 'jpeg2000', 'jpeg']:
        codec_data = eurosat.get(codec, {})
        bpp_list = []
        acc_list = []
        class_acc_dict = {}
        for i in range(1, 7):
            q = f'q{i}'
            entry = codec_data.get(q, {})
            bpp_list.append(entry.get('bpp', entry.get('measured_bpp', entry.get('target_bpp', None))))
            acc_list.append(entry.get('accuracy', None))
            class_acc_dict[q] = entry.get('class_accuracies', {})
        label = codec.upper() if codec != 'jpegai' else 'JPEG-AI'
        data[label] = {
            'bpp': bpp_list,
            'accuracy': acc_list,
            'class_acc': class_acc_dict,
            'type': {
                'jpegai': 'Neural (ISO Standard)',
                'msillm': 'Neural (GAN-based)',
                'cheng2020': 'Neural (MSE-based)',
                'jpeg2000': 'Traditional (Wavelet)',
                'jpeg': 'Traditional (DCT)'
            }[codec],
            'color': {
                'jpegai': '#e41a1c',
                'msillm': '#377eb8',
                'cheng2020': '#4daf4a',
                'jpeg2000': '#984ea3',
                'jpeg': '#ff7f00'
            }[codec],
            'marker': {
                'jpegai': 'o',
                'msillm': 's',
                'cheng2020': '^',
                'jpeg2000': 'D',
                'jpeg': 'v'
            }[codec]
        }
    return data

def plot_rate_distortion(data, baseline_acc=97.65):
    """Plot Rate-Distortion curves (BPP vs Accuracy)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for codec, d in data.items():
        ax.plot(d['bpp'], d['accuracy'], 
                marker=d['marker'], 
                color=d['color'],
                linewidth=2,
                markersize=8,
                label=f"{codec} ({d['type']})")
    
    # Baseline
    ax.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=1.5, 
               label=f'Baseline ResNet18 ({baseline_acc}%)')
    
    ax.set_xlabel('Bits per Pixel (BPP)', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)
    ax.set_title('Rate-Accuracy Trade-off: Neural vs Traditional Codecs\non EuroSAT Satellite Image Classification', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results/codec_comparison_rate_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/codec_comparison_rate_accuracy.pdf', bbox_inches='tight')
    print("Saved: results/codec_comparison_rate_accuracy.png/pdf")
    plt.close()

def plot_rate_distortion_log(data, baseline_acc=97.65):
    """Plot Rate-Distortion curves with log scale for BPP"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for codec, d in data.items():
        ax.semilogx(d['bpp'], d['accuracy'], 
                    marker=d['marker'], 
                    color=d['color'],
                    linewidth=2,
                    markersize=8,
                    label=f"{codec}")
    
    # Baseline
    ax.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=1.5, 
               label=f'Baseline ({baseline_acc}%)')
    
    ax.set_xlabel('Bits per Pixel (BPP) - Log Scale', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)
    ax.set_title('Rate-Accuracy Trade-off (Log Scale)\nLower BPP = Higher Compression', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results/codec_comparison_rate_accuracy_log.png', dpi=300, bbox_inches='tight')
    print("Saved: results/codec_comparison_rate_accuracy_log.png")
    plt.close()

def plot_class_heatmap(data, classes):
    """Plot class-specific accuracy heatmap for lowest quality"""
    codecs_with_class = ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000']
    
    # Create matrix for q1 (lowest quality)
    matrix = np.zeros((len(classes), len(codecs_with_class)))
    
    for j, codec in enumerate(codecs_with_class):
        class_acc = data[codec]['class_acc']['q1']
        for i, cls in enumerate(classes):
            matrix[i, j] = class_acc.get(cls, 0)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(codecs_with_class)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(codecs_with_class, fontsize=12)
    ax.set_yticklabels(classes, fontsize=11)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(codecs_with_class)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('Class-Specific Accuracy at Lowest Quality (q1)\nRed = Poor, Green = Good', fontsize=14)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/class_accuracy_heatmap_q1.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/class_accuracy_heatmap_q1.pdf', bbox_inches='tight')
    print("Saved: results/class_accuracy_heatmap_q1.png/pdf")
    plt.close()

def plot_class_heatmap_q6(data, classes):
    """Plot class-specific accuracy heatmap for highest quality"""
    codecs_with_class = ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000']
    
    # Create matrix for q6 (highest quality)
    matrix = np.zeros((len(classes), len(codecs_with_class)))
    
    for j, codec in enumerate(codecs_with_class):
        class_acc = data[codec]['class_acc']['q6']
        for i, cls in enumerate(classes):
            matrix[i, j] = class_acc.get(cls, 0)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(codecs_with_class)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(codecs_with_class, fontsize=12)
    ax.set_yticklabels(classes, fontsize=11)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(codecs_with_class)):
            val = matrix[i, j]
            color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('Class-Specific Accuracy at Highest Quality (q6)\nRed = Poor, Green = Good', fontsize=14)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/class_accuracy_heatmap_q6.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/class_accuracy_heatmap_q6.pdf', bbox_inches='tight')
    print("Saved: results/class_accuracy_heatmap_q6.png/pdf")
    plt.close()

def plot_codec_bars(data, baseline_acc=97.65):
    """Bar chart comparing codecs at different quality levels with BPP info"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    codecs = ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000', 'JPEG']
    
    # Get BPP values for q1 and q6
    q1_bpp = [data[c]['bpp'][0] for c in codecs]
    q6_bpp = [data[c]['bpp'][5] for c in codecs]
    
    # Create labels with BPP
    q1_labels = [f"{c}\n({q1_bpp[i]:.2f} bpp)" for i, c in enumerate(codecs)]
    q6_labels = [f"{c}\n({q6_bpp[i]:.2f} bpp)" for i, c in enumerate(codecs)]
    
    # Q1 (lowest quality)
    ax1 = axes[0]
    q1_acc = []
    for codec in codecs:
        if codec == 'JPEG':
            q1_acc.append(data[codec]['accuracy'][0])  # quality 10
        else:
            q1_acc.append(data[codec]['accuracy'][0])  # q1
    
    bars1 = ax1.bar(q1_labels, q1_acc, color=[data[c]['color'] for c in codecs])
    ax1.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Lowest Quality Level (q1)', fontsize=13)
    ax1.set_ylim(0, 115)
    ax1.tick_params(axis='x', rotation=0)
    
    # Add value labels
    for bar, val in zip(bars1, q1_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Q6 (highest quality)
    ax2 = axes[1]
    q6_acc = []
    for codec in codecs:
        if codec == 'JPEG':
            q6_acc.append(data[codec]['accuracy'][5])  # quality 95
        else:
            q6_acc.append(data[codec]['accuracy'][5])  # q6
    
    bars2 = ax2.bar(q6_labels, q6_acc, color=[data[c]['color'] for c in codecs])
    ax2.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=1.5, label='Baseline')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Highest Quality Level (q6)', fontsize=13)
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis='x', rotation=0)
    
    # Add value labels
    for bar, val in zip(bars2, q6_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Codec Comparison: Low vs High Quality\n(with Bits per Pixel)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/codec_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/codec_comparison_bars.pdf', bbox_inches='tight')
    print("Saved: results/codec_comparison_bars.png/pdf")
    plt.close()

def plot_individual_codec_curves(data, baseline_acc=97.65):
    """Create individual BPP vs Accuracy plots for each codec"""
    
    codecs = ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000', 'JPEG']
    
    for codec in codecs:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        d = data[codec]
        bpp = d['bpp']
        acc = d['accuracy']
        color = d['color']
        
        # Plot line with markers
        ax.plot(bpp, acc, 'o-', color=color, linewidth=2.5, markersize=10, 
                label=f'{codec}', zorder=3)
        
        # Add labels for each point
        for i, (x, y) in enumerate(zip(bpp, acc)):
            ax.annotate(f'q{i+1}\n({x:.2f} bpp, {y:.1f}%)', 
                       xy=(x, y), xytext=(10, -20), 
                       textcoords='offset points', fontsize=9,
                       ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Baseline
        ax.axhline(y=baseline_acc, color='black', linestyle='--', linewidth=1.5, 
                   label=f'Baseline ({baseline_acc}%)')
        
        # Fill area between curve and baseline
        ax.fill_between(bpp, acc, baseline_acc, alpha=0.2, color=color)
        
        ax.set_xlabel('Bits per Pixel (BPP)', fontsize=12)
        ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
        ax.set_title(f'{codec}: Rate-Accuracy Trade-off\n{d["type"]}', fontsize=13)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Set x-axis based on codec's BPP range
        x_margin = (max(bpp) - min(bpp)) * 0.15
        ax.set_xlim(max(0, min(bpp) - x_margin), max(bpp) + x_margin)
        
        plt.tight_layout()
        
        # Save with codec name (lowercase, no spaces)
        filename = codec.lower().replace('-', '_').replace(' ', '_')
        plt.savefig(f'results/rate_accuracy_{filename}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'results/rate_accuracy_{filename}.pdf', bbox_inches='tight')
        print(f"Saved: results/rate_accuracy_{filename}.png/pdf")
        plt.close()

def generate_latex_table(data, baseline_acc=97.65):
    """Generate LaTeX table for thesis"""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Classification Accuracy (\%) at Different Quality Levels}
\label{tab:codec_comparison}
\begin{tabular}{l|c|cccccc}
\toprule
\textbf{Codec} & \textbf{Type} & \textbf{q1} & \textbf{q2} & \textbf{q3} & \textbf{q4} & \textbf{q5} & \textbf{q6} \\
\midrule
"""
    
    for codec in ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000', 'JPEG']:
        d = data[codec]
        codec_type = d['type'].split('(')[1].replace(')', '') if '(' in d['type'] else d['type']
        accs = ' & '.join([f'{a:.1f}' for a in d['accuracy']])
        latex += f"{codec} & {codec_type} & {accs} \\\\\n"
    
    latex += r"""\midrule
Baseline & ResNet18 & \multicolumn{6}{c}{97.65} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open('results/codec_comparison_table.tex', 'w') as f:
        f.write(latex)
    print("Saved: results/codec_comparison_table.tex")
    
    return latex

def generate_bpp_table(data):
    """Generate table with BPP values"""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Bits per Pixel (BPP) at Different Quality Levels}
\label{tab:bpp_comparison}
\begin{tabular}{l|cccccc}
\toprule
\textbf{Codec} & \textbf{q1} & \textbf{q2} & \textbf{q3} & \textbf{q4} & \textbf{q5} & \textbf{q6} \\
\midrule
"""
    
    for codec in ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000', 'JPEG']:
        d = data[codec]
        bpps = ' & '.join([f'{b:.3f}' for b in d['bpp']])
        latex += f"{codec} & {bpps} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open('results/bpp_comparison_table.tex', 'w') as f:
        f.write(latex)
    print("Saved: results/bpp_comparison_table.tex")

def print_summary(data, baseline_acc=97.65):
    """Print comprehensive summary"""
    
    print("\nCOMPREHENSIVE CODEC COMPARISON - SUMMARY FOR THESIS")
    
    print("\nACCURACY AT EXTREME QUALITY LEVELS:")
    print("-" * 60)
    print(f"{'Codec':<15} {'q1 (Low)':<15} {'q6 (High)':<15} {'Δ to Baseline':<15}")
    print("-" * 60)
    
    for codec in ['JPEG-AI', 'MS-ILLM', 'Cheng2020', 'JPEG2000', 'JPEG']:
        d = data[codec]
        q1 = d['accuracy'][0]
        q6 = d['accuracy'][5]
        delta = baseline_acc - q6
        print(f"{codec:<15} {q1:<15.2f} {q6:<15.2f} {delta:+.2f}%")
    
    print("\n🏆 KEY FINDINGS:")
    print("-" * 60)
    
    # Best at low bitrate
    q1_accs = {c: data[c]['accuracy'][0] for c in data}
    best_q1 = max(q1_accs, key=q1_accs.get)
    print(f"Best at LOW quality (q1): {best_q1} ({q1_accs[best_q1]:.2f}%)")
    
    # Best at high bitrate
    q6_accs = {c: data[c]['accuracy'][5] for c in data}
    best_q6 = max(q6_accs, key=q6_accs.get)
    print(f"Best at HIGH quality (q6): {best_q6} ({q6_accs[best_q6]:.2f}%)")
    
    # Biggest improvement q1→q6
    improvements = {c: data[c]['accuracy'][5] - data[c]['accuracy'][0] for c in data}
    best_improve = max(improvements, key=improvements.get)
    print(f"Biggest improvement q1→q6: {best_improve} (+{improvements[best_improve]:.2f}%)")
    
    # JPEG-AI ceiling problem
    print(f"\nJPEG-AI CEILING PROBLEM:")
    print(f"   - Even at highest quality (q6, 1.5 bpp): {data['JPEG-AI']['accuracy'][5]:.2f}%")
    print(f"   - Still {baseline_acc - data['JPEG-AI']['accuracy'][5]:.2f}% below baseline!")
    print(f"   - MS-ILLM at q6 (0.9 bpp): {data['MS-ILLM']['accuracy'][5]:.2f}% (only {baseline_acc - data['MS-ILLM']['accuracy'][5]:.2f}% below)")
    
    print("\n📝 THESIS IMPLICATIONS:")
    print("-" * 60)
    print("1. GAN-based codecs (MS-ILLM) excel at HIGH bitrates")
    print("2. JPEG-AI competitive at LOW bitrates despite tools_off config")
    print("3. MSE-based neural (Cheng2020) worst performer overall")
    print("4. Traditional JPEG2000 needs HIGH bitrate to be useful")
    print("5. JPEG-AI has permanent ~10% accuracy ceiling (no enhancement filters)")

def main():
    print("Loading results...")
    results = load_results()
    
    print("Extracting data...")
    data = extract_data(results)
    
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
               'River', 'SeaLake']
    
    print("\nGenerating visualizations...")
    plot_rate_distortion(data)
    plot_rate_distortion_log(data)
    plot_class_heatmap(data, classes)
    plot_class_heatmap_q6(data, classes)
    plot_codec_bars(data)
    plot_individual_codec_curves(data)
    
    print("\nGenerating LaTeX tables...")
    generate_latex_table(data)
    generate_bpp_table(data)
    
    print_summary(data)
    
    print("\nAll visualizations and tables generated!")
    print("   Check the 'results/' folder for output files.")

if __name__ == '__main__':
    main()
