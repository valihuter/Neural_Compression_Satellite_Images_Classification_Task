# Scripts Documentation

This directory contains all scripts used for image compression, evaluation, and visualization for the Master's thesis on neural image compression for satellite imagery.

## Directory Structure

```
scripts/
├── compression/          # Compress images with different codecs (10 scripts)
├── evaluation/           # Evaluate classifiers on compressed images (2 scripts)
└── visualization/        # Generate figures for thesis (5 scripts)
```

---

## Compression Scripts

**Location:** `scripts/compression/`

Compress satellite images using 5 codecs at 6 quality levels each.

### EuroSAT Compression (64×64 pixels, 10 classes)

| Script | Codec | Quality Levels | BPP Range | Hardware |
|--------|-------|----------------|-----------|----------|
| `01_eurosat_jpeg_compress.py` | JPEG | q1-q6 (10-95) | 0.45-1.73 | CPU (Pillow) |
| `02_eurosat_jpeg2000_compress.py` | JPEG2000 | q1-q6 (CR 5-160) | 0.10-0.97 | CPU (OpenJPEG) |
| `03_eurosat_cheng2020_compress.py` | Cheng2020 | q1-q6 | 0.13-0.72 | GPU (CompressAI) |
| `04_eurosat_msillm_compress.py` | MS-ILLM | q1-q6 | 0.04-0.90 | GPU (CompressAI) |
| `05_eurosat_jpegai_compress.py` | JPEG-AI | q1-q6 | 0.13-1.50 | GPU (ISO RefSW) |

### RESISC45 Compression (256×256 pixels, 11 classes)

| Script | Codec | Quality Levels | BPP Range | Hardware |
|--------|-------|----------------|-----------|----------|
| `06_resisc45_jpeg_compress.py` | JPEG | q1-q6 (10-95) | 0.30-2.23 | CPU (Pillow) |
| `07_resisc45_jpeg2000_compress.py` | JPEG2000 | q1-q6 (CR 5-160) | 0.24-2.39 | CPU (OpenJPEG) |
| `08_resisc45_cheng2020_compress.py` | Cheng2020 | q3-q6 | 0.30-0.98 | GPU (CompressAI) |
| `09_resisc45_msillm_compress.py` | MS-ILLM | q1-q6 | 0.04-0.81 | GPU (CompressAI) |
| `10_resisc45_jpegai_compress.py` | JPEG-AI | q1-q6 | 0.12-1.50 | CPU (8 workers) |

**Usage:**
```bash
# Compress EuroSAT with all codecs
python scripts/compression/01_eurosat_jpeg_compress.py
python scripts/compression/02_eurosat_jpeg2000_compress.py
python scripts/compression/03_eurosat_cheng2020_compress.py
python scripts/compression/04_eurosat_msillm_compress.py
python scripts/compression/05_eurosat_jpegai_compress.py

# Compress RESISC45 with all codecs
python scripts/compression/06_resisc45_jpeg_compress.py
python scripts/compression/07_resisc45_jpeg2000_compress.py
python scripts/compression/08_resisc45_cheng2020_compress.py
python scripts/compression/09_resisc45_msillm_compress.py
python scripts/compression/10_resisc45_jpegai_compress.py
```

**Output:**
- Compressed images: `data/{dataset}/comp/{codec}/q{1-6}/{class}/{image}.png`
- BPP measurements: Calculated automatically from file size

---

## Evaluation Scripts

**Location:** `scripts/evaluation/`

Evaluate trained classifiers (ResNet-18, ViT-S/16) on compressed images.

| Script | Dataset | Models | Codecs | Test Images |
|--------|---------|--------|--------|-------------|
| `01_eurosat_evaluate_all_codecs.py` | EuroSAT | ResNet-18, ViT-S/16 | All 5 × 6 levels | 4,050 |
| `02_resisc45_evaluate_all_codecs.py` | RESISC45 | ResNet-18, ViT-S/16 | All 5 × 6 levels | 1,155 |

**Usage:**
```bash
# Evaluate all codecs on EuroSAT (30-60 min)
python scripts/evaluation/01_eurosat_evaluate_all_codecs.py

# Evaluate all codecs on RESISC45 (1-2 hours)
python scripts/evaluation/02_resisc45_evaluate_all_codecs.py
```

**Output:**
- `results/json/{dataset}_{codec}_evaluation.json`
- Metrics: Accuracy, per-class accuracy, BPP (mean/std), PSNR, MS-SSIM

---

## Visualization Scripts

**Location:** `scripts/visualization/`

Generate all figures used in the thesis.

| # | Script | Thesis Figures | Description |
|---|--------|----------------|-------------|
| 01 | `01_eurosat_class_examples.py` | `eurosat_class_examples.pdf`<br>`miscompression_examples.pdf` | Class grid (2×5) + compression comparison for Forest class |
| 02 | `02_misclassification_examples.py` | `eurosat_misclassification_examples.pdf`<br>`resisc45_misclassification_detailed.pdf` | Compression-induced misclassification patterns |
| 03 | `03_rate_accuracy_plots.py` | `codec_comparison_rate_accuracy.pdf`<br>`class_accuracy_heatmap_q6.pdf` | Main rate-accuracy curves + per-class heatmaps |
| 04 | `04_vit_vs_resnet_comparison.py` | `vit_vs_resnet_rate_accuracy.pdf` | Architecture robustness comparison |
| 05 | `05_resisc45_analysis_plots.py` | `resisc45_rate_accuracy_comparison.pdf`<br>`resisc45_jpegai_class_heatmap.pdf`<br>`cross_dataset_jpegai_comparison.pdf` | RESISC45-specific analysis |

**Usage:**
```bash
# Generate all thesis figures (5-10 min total)
python scripts/visualization/01_eurosat_class_examples.py
python scripts/visualization/02_misclassification_examples.py
python scripts/visualization/03_rate_accuracy_plots.py
python scripts/visualization/04_vit_vs_resnet_comparison.py
python scripts/visualization/05_resisc45_analysis_plots.py
```

**Output:**
- Thesis PDFs: `docs/thesis_fhkufstein/img/*.pdf`
- Result PNGs: `results/*.png`

---

## Complete Workflow

### 1. Data Preparation
```bash
# Download datasets
# - EuroSAT RGB: https://github.com/phelber/EuroSAT
# - RESISC45: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

# Place in:
# - data/EUROSAT/uncomp/EuroSAT_RGB/
# - data/RESISC45_SUBSET11/UNCOMP/NWPU-RESISC45/
```

### 2. Train Classifiers (Google Colab)
```bash
# Train ResNet-18 and ViT-S/16 on both datasets
# See: models/EUROSAT/train_*.ipynb
# See: models/RESISC45_SUBSET11/RESISC45_SUBSET_Training/train_models.py
```

### 3. Compression (1-12 hours depending on codec/hardware)
```bash
# EuroSAT
for i in {01..05}; do
    python scripts/compression/${i}_eurosat*.py
done

# RESISC45
for i in {06..10}; do
    python scripts/compression/${i}_resisc45*.py
done
```

### 4. Evaluation (2-4 hours total)
```bash
python scripts/evaluation/01_eurosat_evaluate_all_codecs.py
python scripts/evaluation/02_resisc45_evaluate_all_codecs.py
```

### 5. Visualization (5-10 minutes)
```bash
for script in scripts/visualization/*.py; do
    python "$script"
done
```

---

## Requirements

### Python Dependencies
```bash
pip install torch torchvision
pip install compressai
pip install pillow opencv-python
pip install matplotlib seaborn
pip install numpy pandas scipy
pip install tqdm
```

### External Dependencies

**JPEG2000 (OpenJPEG):**
```bash
# macOS
brew install openjpeg

# Ubuntu
sudo apt-get install libopenjp2-tools
```

**JPEG-AI:**
- Download ISO reference software: https://jpeg.org/jpegai/software.html
- Place in `third_party/jpegai/`

### Hardware Recommendations

| Task | Minimum | Recommended | Time (Recommended) |
|------|---------|-------------|---------------------|
| JPEG/JPEG2000 | CPU | CPU | 30 min |
| Cheng2020/MS-ILLM | GTX 1060 | RTX 3090+ | 2-4 hours |
| JPEG-AI | CPU (8 cores) | RTX 3090 | 12-24 hours |
| Evaluation | GTX 1060 | RTX 3090+ | 2-4 hours |
| Visualization | CPU | CPU | 10 minutes |

---

## Naming Convention

All scripts follow consistent naming:

**Pattern:** `{number}_{dataset}_{codec}_compress.py` or `{number}_{task}.py`

Where:
- `{number}`: Execution order (01-10 for compression, 01-05 for visualization)
- `{dataset}`: `eurosat` or `resisc45`
- `{codec}`: `jpeg`, `jpeg2000`, `cheng2020`, `msillm`, or `jpegai`
- `{task}`: `evaluate_all_codecs`, `class_examples`, `misclassification_examples`, etc.

**Examples:**
- `01_eurosat_jpeg_compress.py` → Compress EuroSAT with JPEG
- `03_rate_accuracy_plots.py` → Generate rate-accuracy curves for thesis
- `10_resisc45_jpegai_compress.py` → Compress RESISC45 with JPEG-AI

---

## Quality Level Mapping

Each codec has 6 quality levels (q1-q6) mapping to different internal parameters:

| Codec | q1 | q2 | q3 | q4 | q5 | q6 | Parameter Type |
|-------|----|----|----|----|----|----|----------------|
| JPEG | 10 | 25 | 50 | 75 | 90 | 95 | Quality (1-100) |
| JPEG2000 | 5 | 10 | 20 | 40 | 80 | 160 | Compression Ratio |
| Cheng2020 | 1 | 2 | 3 | 4 | 5 | 6 | Lambda (MSE) |
| MS-ILLM | 1 | 2 | 3 | 4 | 5 | 6 | Quality Index |
| JPEG-AI | 1 | 2 | 3 | 4 | 5 | 6 | Quality Parameter |

### Measured BPP Ranges

**EuroSAT (64×64):**
| Codec | Min BPP (q1) | Max BPP (q6) |
|-------|--------------|--------------|
| JPEG | 0.45 | 1.73 |
| JPEG2000 | 0.10 | 0.97 |
| Cheng2020 | 0.13 | 0.72 |
| MS-ILLM | 0.04 | 0.90 |
| JPEG-AI | 0.13 | 1.50 |

**RESISC45 (256×256):**
| Codec | Min BPP (q1) | Max BPP (q6) |
|-------|--------------|--------------|
| JPEG | 0.30 | 2.23 |
| JPEG2000 | 0.24 | 2.39 |
| Cheng2020 | 0.30 | 0.98 |
| MS-ILLM | 0.04 | 0.81 |
| JPEG-AI | 0.12 | 1.50 |

---

## Archived Scripts

**Location:** `archive/scripts_unused/`

Scripts moved to archive (not used in final thesis):
- `generate_class_plots.py` - Superseded by `03_rate_accuracy_plots.py`
- `generate_resisc45_plots.py` - Superseded by `05_resisc45_analysis_plots.py`
- `generate_vit_resnet_comparison.py` - Superseded by `04_vit_vs_resnet_comparison.py`

These scripts generated redundant or alternative visualizations that were not included in the final thesis.

---

## Reproducibility

### Random Seeds
All experiments use **seed=42** for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Data Splits
Train/Val/Test splits (70/15/15, stratified) stored in:
- `data/EUROSAT/uncomp/splits/*.json`
- `data/RESISC45_SUBSET11/UNCOMP/splits/*.json`

### Trained Models
Classifier weights stored in:
- `models/EUROSAT/*.pth`
- `models/RESISC45_SUBSET11/*.pth`

### Results Consolidation
All results consolidated in:
- `results/json/master_results.json`

---

## Support

For questions about scripts or reproduction:
1. Check script docstrings and inline comments
2. Review thesis methodology: `docs/thesis_fhkufstein/chapters/chap04.tex`
3. Consult master results: `results/json/master_results.json`
4. Contact repository maintainer

---

## Important Notes

### JPEG-AI Hardware Differences
- **EuroSAT (64×64):** GPU mode with sequential processing (requires upscaling from 64→128)
- **RESISC45 (256×256):** CPU mode with 8 parallel workers (native resolution support)

### Cheng2020 Quality Levels
- **EuroSAT:** All 6 quality levels (q1-q6)
- **RESISC45:** Only q3-q6 (q1-q2 too aggressive for 256×256 images)

### Compression Time Estimates
- **JPEG/JPEG2000:** 1-10 ms/image (CPU)
- **Cheng2020/MS-ILLM:** 50-200 ms/image (GPU)
- **JPEG-AI:** 500-2000 ms/image (GPU sequential or CPU parallel)

Total compression time: **8-16 hours** for all codecs on both datasets (with recommended hardware).
