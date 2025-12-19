#!/usr/bin/env python3
"""
Calculate Bjøntegaard Delta Rate (BD-Rate) for codec comparison.

BD-Rate is the standard metric for comparing video/image codecs.
It measures the average bitrate difference between two codecs
at the same quality level (or vice versa).

References:
- Bjøntegaard, G. (2001). Calculation of average PSNR differences between RD-curves.
  ITU-T SG16 Doc. VCEG-M33.
- Bjøntegaard, G. (2008). Improvements of the BD-PSNR model.
  ITU-T SG16 Doc. VCEG-AI11.

For classification accuracy (instead of PSNR), we adapt the method:
- Negative BD-Rate = codec is better (needs fewer bits for same accuracy)
- Positive BD-Rate = codec is worse (needs more bits for same accuracy)
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import json
from pathlib import Path

def bd_rate(rate1, quality1, rate2, quality2):
    """
    Calculate BD-Rate between two codecs.
    
    Args:
        rate1: Bitrates (BPP) for reference codec
        quality1: Quality metric (accuracy) for reference codec
        rate2: Bitrates (BPP) for test codec
        quality2: Quality metric (accuracy) for test codec
    
    Returns:
        BD-Rate in percentage. Negative means test codec is better.
    """
    # Need at least 4 points for cubic interpolation
    if len(rate1) < 4 or len(rate2) < 4:
        return bd_rate_linear(rate1, quality1, rate2, quality2)
    
    # Log scale for rate
    log_rate1 = np.log10(rate1)
    log_rate2 = np.log10(rate2)
    
    # Find overlapping quality range
    min_quality = max(min(quality1), min(quality2))
    max_quality = min(max(quality1), max(quality2))
    
    if min_quality >= max_quality:
        return float('nan')
    
    # Fit polynomial: log(rate) = f(quality)
    try:
        poly1 = np.polyfit(quality1, log_rate1, 3)
        poly2 = np.polyfit(quality2, log_rate2, 3)
    except np.linalg.LinAlgError:
        return float('nan')
    
    # Integrate over quality range
    def integrate_poly(poly, low, high):
        anti = np.polyint(poly)
        return np.polyval(anti, high) - np.polyval(anti, low)
    
    int1 = integrate_poly(poly1, min_quality, max_quality)
    int2 = integrate_poly(poly2, min_quality, max_quality)
    
    avg_diff = (int2 - int1) / (max_quality - min_quality)
    bd_rate_value = (10 ** avg_diff - 1) * 100
    
    return bd_rate_value

def bd_rate_linear(rate1, quality1, rate2, quality2):
    """Simplified BD-Rate using linear interpolation for fewer data points."""
    min_quality = max(min(quality1), min(quality2))
    max_quality = min(max(quality1), max(quality2))
    
    if min_quality >= max_quality:
        return float('nan')
    
    try:
        f1 = interpolate.interp1d(quality1, np.log10(rate1), 
                                   kind='linear', fill_value='extrapolate')
        f2 = interpolate.interp1d(quality2, np.log10(rate2), 
                                   kind='linear', fill_value='extrapolate')
    except ValueError:
        return float('nan')
    
    quality_samples = np.linspace(min_quality, max_quality, 100)
    log_rate1_samples = f1(quality_samples)
    log_rate2_samples = f2(quality_samples)
    avg_diff = np.mean(log_rate2_samples - log_rate1_samples)
    bd_rate_value = (10 ** avg_diff - 1) * 100
    
    return bd_rate_value

def interpolate_accuracy_at_bpp(bpp_values, accuracy_values, target_bpp):
    """Interpolate accuracy at specific BPP values."""
    sorted_idx = np.argsort(bpp_values)
    bpp_sorted = np.array(bpp_values)[sorted_idx]
    acc_sorted = np.array(accuracy_values)[sorted_idx]
    
    f = interpolate.interp1d(bpp_sorted, acc_sorted, kind='linear',
                              bounds_error=False, fill_value=np.nan)
    return f(target_bpp)

def generate_latex_table(codec_data, target_bpp, codecs):
    """Generate LaTeX table for BPP-based comparison."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Classification accuracy (\%) at standardized bitrates. Values are linearly interpolated from measured data points. Cells marked with ``---'' indicate bitrates outside the codec's operating range.}
\label{tab:accuracy-at-bpp}
\begin{tabular}{l""" + "c" * len(codecs) + r"""}
\toprule
"""
    
    latex += "BPP"
    for codec in codecs:
        latex += f" & {codec}"
    latex += r" \\" + "\n" + r"\midrule" + "\n"
    
    # ResNet section
    latex += r"\multicolumn{" + str(len(codecs) + 1) + r"}{l}{\textit{ResNet-18}} \\" + "\n"
    
    for bpp in target_bpp:
        latex += f"{bpp:.2f}"
        for codec in codecs:
            data = codec_data[codec]
            acc = interpolate_accuracy_at_bpp(data['bpp'], data['resnet_acc'], [bpp])[0]
            if np.isnan(acc):
                latex += " & ---"
            else:
                latex += f" & {acc:.1f}"
        latex += r" \\" + "\n"
    
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{" + str(len(codecs) + 1) + r"}{l}{\textit{ViT-S/16}} \\" + "\n"
    
    for bpp in target_bpp:
        latex += f"{bpp:.2f}"
        for codec in codecs:
            data = codec_data[codec]
            acc = interpolate_accuracy_at_bpp(data['bpp'], data['vit_acc'], [bpp])[0]
            if np.isnan(acc):
                latex += " & ---"
            else:
                latex += f" & {acc:.1f}"
        latex += r" \\" + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex

def main():
    results_dir = Path('/Users/vali/MA-Neural_Compression_Satellite_Images/results')
    csv_path = results_dir / 'accuracy' / 'vit_vs_resnet_full_comparison.csv'
    
    df = pd.read_csv(csv_path)
    codecs = df['Codec'].unique()
    
    print("BD-Rate Analysis: Codec Comparison for Satellite Image Classification")
    print()
    print("Reference: Bjøntegaard, G. (2001). Calculation of average PSNR differences")
    print("           between RD-curves. ITU-T SG16 Doc. VCEG-M33.")
    print()
    print("Interpretation:")
    print("  - Negative BD-Rate = Test codec is BETTER (fewer bits for same accuracy)")
    print("  - Positive BD-Rate = Test codec is WORSE (more bits for same accuracy)")
    print()
    
    codec_data = {}
    for codec in codecs:
        codec_df = df[df['Codec'] == codec].sort_values('BPP')
        codec_data[codec] = {
            'bpp': codec_df['BPP'].values,
            'vit_acc': codec_df['ViT_Accuracy'].values,
            'resnet_acc': codec_df['ResNet_Accuracy'].values
        }
    
    reference = 'JPEG'
    
    print(f"BD-Rate vs. {reference} (Reference Codec)")
    
    bd_results = {'reference': reference, 'comparisons': {}}
    
    for classifier in ['ResNet', 'ViT']:
        acc_key = 'resnet_acc' if classifier == 'ResNet' else 'vit_acc'
        print(f"\n{classifier} Classifier:")
        print(f"{'Codec':<15} {'BD-Rate':>12} {'Interpretation':<30}")
        
        ref_data = codec_data[reference]
        
        for codec in codecs:
            if codec == reference:
                print(f"{codec:<15} {'---':>12} (reference)")
                continue
            
            test_data = codec_data[codec]
            bd = bd_rate(ref_data['bpp'], ref_data[acc_key],
                        test_data['bpp'], test_data[acc_key])
            
            if np.isnan(bd):
                interpretation = "insufficient overlap"
            elif bd < -20:
                interpretation = "significantly better"
            elif bd < -5:
                interpretation = "better"
            elif bd < 5:
                interpretation = "comparable"
            elif bd < 20:
                interpretation = "worse"
            else:
                interpretation = "significantly worse"
            
            print(f"{codec:<15} {bd:>+11.1f}% {interpretation:<30}")
            
            if classifier not in bd_results['comparisons']:
                bd_results['comparisons'][classifier] = {}
            bd_results['comparisons'][classifier][codec] = {
                'bd_rate': round(bd, 2) if not np.isnan(bd) else None,
                'interpretation': interpretation
            }
    
    output_path = results_dir / 'codec_comparison' / 'bd_rate_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(bd_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\nAccuracy at Standard BPP Points")
    
    target_bpp = [0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
    
    for classifier in ['ResNet', 'ViT']:
        acc_key = 'resnet_acc' if classifier == 'ResNet' else 'vit_acc'
        print(f"\n{classifier} Accuracy (%) at different bitrates:")
        print()
        
        header = f"{'BPP':<8}"
        for codec in codecs:
            header += f"{codec:>12}"
        print(header)
        
        table_data = []
        for bpp in target_bpp:
            row = {'BPP': bpp}
            row_str = f"{bpp:<8.2f}"
            
            for codec in codecs:
                data = codec_data[codec]
                acc = interpolate_accuracy_at_bpp(data['bpp'], data[acc_key], [bpp])[0]
                
                if np.isnan(acc):
                    row_str += f"{'---':>12}"
                    row[codec] = None
                else:
                    row_str += f"{acc:>11.1f}%"
                    row[codec] = round(acc, 2)
            
            print(row_str)
            table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        csv_output = results_dir / 'accuracy' / f'accuracy_at_bpp_{classifier.lower()}.csv'
        table_df.to_csv(csv_output, index=False)
        print(f"\nSaved to: {csv_output}")
    
    print("\nLaTeX Table (for thesis)")
    
    latex_output = generate_latex_table(codec_data, target_bpp, codecs)
    print(latex_output)
    
    latex_path = results_dir / 'bpp' / 'bpp_comparison_table_interpolated.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_output)
    print(f"\nSaved to: {latex_path}")

if __name__ == '__main__':
    main()
