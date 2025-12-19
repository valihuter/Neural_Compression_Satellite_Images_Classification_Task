# Work Protocol: Neural Compression for Satellite Images

**Student:** Vali Huter  
**Project:** Master's Thesis - Neural Compression for Satellite Images
**Repository:** MA-Neural_Compression_Satellite_Images
**Last Updated:** December 16, 2025

---

## Research Questions

1. **RQ1:** How does neural image compression affect classification accuracy of satellite images compared to traditional codecs?

2. **RQ2:** At which bitrates do neural methods outperform traditional compression?

3. **RQ3:** Which land-use classes are particularly susceptible to compression artifacts?

4. **RQ4:** What types of semantic changes occur in satellite imagery after neural compression, and how do they correlate with classification errors?

---

## Datasets

### EuroSAT RGB

| Property | Value |
|----------|-------|
| Dataset | EuroSAT RGB (Helber et al., 2019) |
| Source | Sentinel-2 satellite imagery |
| Total images | 27,000 |
| Resolution | 64×64 pixels |
| Channels | 3 (RGB: B4, B3, B2) |
| GSD | 10 meters/pixel |
| Classes | 10 land cover types |
| Train split | 18,900 images (70%) |
| Val split | 4,050 images (15%) |
| Test split | 4,050 images (15%) |
| **Source format** | **JPEG (3-9 BPP, high quality)** |
| Random seed | 42 (reproducibility) |

**Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

**Methodological Note on Source Format:**
> The EuroSAT dataset is distributed as high-quality JPEG images (5-9 BPP depending on class complexity). Classification models are thus trained on already JPEG-encoded images. This reflects realistic application scenarios where training data rarely exists as uncompressed raw data. The high source quality (5-9 BPP vs. our lowest test level at ~0.1 BPP) minimizes potential bias in favor of JPEG.

### RESISC45 Subset (NEW - Added Dec 2025)

| Property | Value |
|----------|-------|
| Dataset | NWPU-RESISC45 subset (Cheng et al., 2017) |
| Source | Google Earth imagery |
| Total images | 7,700 (11 classes × 700 images) |
| Resolution | 256×256 pixels |
| Channels | 3 (RGB) |
| GSD | 0.2-30 meters/pixel (variable) |
| Classes | 11 scene categories |
| Train split | 5,390 images (70%) |
| Val split | 1,155 images (15%) |
| Test split | 1,155 images (15%) |
| **Source format** | **JPEG (2-8 BPP)** |
| Random seed | 42 (same as EuroSAT) |

**Classes (11 EuroSAT-relevant categories):**
- beach
- circular_farmland
- dense_residential
- forest
- freeway
- industrial_area
- lake
- meadow
- medium_residential
- rectangular_farmland
- river

**Subset Selection Rationale:**
From the full 45-class RESISC45 dataset, we selected 11 classes that provide semantic correspondence to EuroSAT categories while introducing new visual challenges through higher resolution (256×256 vs 64×64). This enables cross-dataset comparison while maintaining manageable computational requirements for a master's thesis scope.

**Key Differences from EuroSAT:**
1. **Resolution:** 4× higher (256×256 vs 64×64) enables examination of compression effects at higher spatial detail
2. **JPEG-AI compatibility:** Native 256×256 meets minimum 128×128 requirement (no upscaling workarounds needed)
3. **Variable GSD:** 0.2-30m heterogeneity tests codec robustness across scales
4. **Source diversity:** Google Earth vs Sentinel-2 ensures findings not sensor-specific

---

## Environment Setup

### Python Environment
- **Version:** 3.9.6 (system default macOS)
- **Environment:** venv (not conda)
  - Reason: Simpler dependency management
- **Key Dependencies:**
  - PyTorch 2.2.2
  - TensorFlow 2.16.2
  - CompressAI 1.2.8
  - torchvision, scikit-image, opencv-python, lpips, pillow
  - timm (for ViT models)

### Hardware Used
- **Local Development:** MacBook Pro (2020, 2GHz Quad-Core Intel Core i5, 8GB RAM)
- **GPU Training:** Google Colab (NVIDIA Tesla T4)
- **GPU Compression:** RunPod Cloud (NVIDIA RTX 3090, RTX 4090)
- **CPU Compression:** RunPod 16 vCPU pods (for JPEG-AI RESISC45)

### Compression Hardware Details

| Codec | EuroSAT (64×64) | RESISC45 (256×256) |
|-------|-----------------|---------------------|
| JPEG | Local Mac CPU | Local Mac CPU |
| JPEG2000 | Local Mac CPU | Local Mac CPU |
| Cheng2020 | RunPod RTX 3090 GPU | RunPod RTX 3090 GPU |
| MS-ILLM | RunPod 2× RTX 4090 GPU | RunPod 2× RTX 4090 GPU |
| JPEG-AI | RunPod RTX 3090 **GPU** (sequential) | RunPod 16 vCPU **CPU** (8 parallel workers) |

**JPEG-AI Note:** For RESISC45, CPU mode with parallel workers was more stable than GPU mode due to C++ extension build issues on some pods. The parallel CPU approach was also faster overall for the larger 256×256 images.

### Software Stack
- Python 3.9.6
- PyTorch 2.2.2
- torchvision
- CompressAI 1.2.8 (neural codecs)
- OpenJPEG (JPEG2000)
- JPEG-AI Official ISO reference software (GitLab)

---

## Classification Models

### ResNet-18 (CNN Baseline)

**Architecture:**
- ResNet-18 pretrained on ImageNet1k
- Parameters: 11.2M
- Final FC layer: 512 → C classes (C=10 for EuroSAT, C=11 for RESISC45)

**Training Configuration:**
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: Cross-entropy
- Batch size: 64
- Max epochs: 20
- LR scheduler: ReduceLROnPlateau (factor=0.1, patience=3)
- Early stopping: Patience 5 epochs
- Data augmentation:
  - RandomHorizontalFlip (p=0.5)
  - RandomVerticalFlip (p=0.5)
  - RandomRotation (±10°)
  - ColorJitter (brightness, contrast, saturation: ±0.2)
  - Rationale: Satellite images have no canonical orientation
- Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Input size: Native resolution (64×64 for EuroSAT, 256×256 for RESISC45)

**Training Results:**

*EuroSAT:*
- Training: Local Mac CPU (~90 minutes, 1.3 it/s)
- Best val accuracy: 97.60% (epoch 18)
- **Test accuracy: 97.65%**
- Training accuracy: 98.11%
- Per-class precision range: 93.9% (PermanentCrop) to 99.8% (Forest, Residential)

*RESISC45:*
- Training: Google Colab T4 GPU (~20-30 minutes)
- Best val accuracy: 97.40% (epoch 20)
- **Test accuracy: 96.54%**
- Training through all epochs (no early stopping triggered)

### ViT-S/16 (Transformer Architecture)

**Architecture:**
- Vision Transformer Small (ViT-S/16)
- Parameters: 22M
- Patch size: 16×16
- Classification head: → C classes (C=10 for EuroSAT, C=11 for RESISC45)
- Source: timm library with ImageNet pretrained weights

**Training Configuration:**
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- LR scheduler: Cosine annealing (T_max=15)
- Loss: Cross-entropy
- Batch size: 64
- Epochs: 15 (no early stopping)
- Data augmentation: Same as ResNet-18
- Normalization: ImageNet stats
- Input size: 224×224 (ViT standard)
  - EuroSAT: Bilinear upsample from 64×64 (3.5× scaling, introduces artifacts)
  - RESISC45: Bilinear downsample from 256×256 (more benign)

**Training Results:**

*EuroSAT:*
- Training: Google Colab T4 GPU (~10-12 minutes)
- **Test accuracy: 99.73%**
- Significantly outperforms ResNet-18 despite resolution mismatch

*RESISC45:*
- Training: Google Colab T4 GPU (~15-20 minutes)
- Best val accuracy: 98.89% (epoch 8)
- **Test accuracy: 98.10%**
- Early convergence (epoch 8) suggests good fit with native 256×256 resolution

**Architecture Comparison:**
- ViT achieves +2.08% higher accuracy on EuroSAT uncompressed
- But: ViT shows lower robustness to compression artifacts (finding from EuroSAT experiments)
- ResNet's local convolutions more robust than ViT's patch-based global attention

---

## Compression Codecs

### Traditional Codecs

**JPEG (DCT-based):**
- Library: PIL/Pillow
- Quality levels: q10, q25, q50, q75, q90, q95
- Characteristics: Block-based DCT, 8×8 blocks, blocking artifacts at low quality
- Status:  Complete for both datasets

**JPEG2000 (Wavelet-based):**
- Library: OpenCV (with OpenJPEG backend)
- Quality levels: q1-q6
- Characteristics: DWT-based, better low-bitrate performance than JPEG
- Remote sensing standard (Sentinel-2 distribution format)
- Status:  Complete for both datasets

### Neural Codecs

**Cheng2020-attn (MSE-optimized):**
- Architecture: Cheng et al. (2020) with attention modules
- Optimization: MSE-based rate-distortion loss
- Quality levels: q1-q6 (lambda values 8→3)
- Library: CompressAI
- Status:  Complete for both datasets
- **Key Finding:** Forest class complete failure (0% accuracy q1-q5) on EuroSAT

**MS-ILLM (GAN-based perceptual):**
- Architecture: Muckley et al. (2023) with GAN discriminator
- Optimization: Perceptual loss + adversarial training
- Quality levels: q1-q6
- Library: torch.hub (facebookresearch/NeuralCompression)
- Status:  Complete for both datasets
- **Key Finding:** Best codec for machine vision (97.75% at 0.9 BPP on EuroSAT)

**JPEG-AI (ISO/IEC 6048-1:2025):**
- First standardized neural codec
- Architecture: Neural autoencoder with learned entropy coding
- Quality levels: q1-q6
- Library: Official ISO reference software (GitLab)
- Minimum resolution: 128×128 (problematic for EuroSAT 64×64)
- Status:  Complete for both datasets
- **EuroSAT Issue:** Ceiling at 88% due to mandatory `tools_off.json` for 64×64
- **RESISC45:** Native 256×256 meets requirements (no upscaling)

---

## Compression Status & Statistics

### EuroSAT RGB (Primary Dataset)

| Codec | Quality Levels | Images/Level | Total Compressed | Status |
|-------|----------------|--------------|------------------|--------|
| JPEG | q10, q25, q50, q75, q90, q95 | 4,050 | 24,300 |  Complete |
| JPEG2000 | q1-q6 | 27,000 | 162,000 |  Complete |
| Cheng2020 | q1-q6 | 4,050 | 24,300 |  Complete |
| MS-ILLM | q1-q6 | 27,000 | 162,000 |  Complete |
| JPEG-AI | q1-q6 | 4,050 | 24,300 |  Complete |
| **Total** | | | **396,900** |  All done |

### RESISC45 Subset (Cross-Dataset Validation)

| Codec | Quality Levels | Images/Level | Total Compressed | Status |
|-------|----------------|--------------|------------------|--------|
| JPEG | q10, q25, q50, q75, q90, q95 | 1,155 | 6,930 |  Complete |
| JPEG2000 | q1-q6 | 7,700 | 46,200 |  Complete |
| Cheng2020 | q3-q6 | 1,155 | 4,620 |  Complete |
| MS-ILLM | q1-q6 | 7,700 | 46,200 |  Complete |
| JPEG-AI | q1-q6 | 7,700 | 46,200 |  Complete |
| **Total** | | | **150,150** |  **All Complete** |

**Note:** Cheng2020 q1-q2 skipped due to quality level mismatch with pretrained models.

---

## Measured Bitrates (BPP) - CORRECTED

### Why BPP Varies by Dataset
The same quality parameters produce different bitrates on different datasets due to:
1. **Image size:** 64×64 (EuroSAT) vs 256×256 (RESISC45) = 16× more pixels
2. **Content complexity:** Different texture/color distributions affect entropy coding
3. **Codec behavior:** Only JPEG-AI is rate-controlled (target BPP = measured BPP)

### EuroSAT Measured BPP

| Codec | q1 | q2 | q3 | q4 | q5 | q6 |
|-------|----|----|----|----|----|----|
| **JPEG** | 0.45 | 0.59 | 0.77 | 0.89 | 1.20 | 1.73 |
| **JPEG2000** | 0.10 | 0.25 | 0.40 | 0.57 | 0.77 | 0.97 |
| **Cheng2020** | 0.13 | 0.18 | 0.26 | 0.39 | 0.53 | 0.72 |
| **MS-ILLM** | 0.04 | 0.07 | 0.14 | 0.28 | 0.56 | 0.90 |
| **JPEG-AI** | 0.13 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 |

**Source:** Logged compression results from `compression_results.json` files.

### RESISC45 Measured BPP

| Codec | q1 | q2 | q3 | q4 | q5 | q6 |
|-------|----|----|----|----|----|----|
| **JPEG** | 0.302 | 0.658 | 1.201 | 1.549 | 1.900 | 2.230 |
| **JPEG2000** | 0.235 | 0.351 | 0.592 | 0.950 | 1.702 | 2.386 |
| **Cheng2020** | N/A | N/A | 0.304 | 0.494 | 0.709 | 0.982 |
| **MS-ILLM** | 0.037 | 0.064 | 0.121 | 0.224 | 0.429 | 0.806 |
| **JPEG-AI** | 0.12 | 0.25 | 0.50 | 0.75 | 1.00 | 1.50 |

**Source:** Extracted from RESISC45 compression results. JPEG-AI uses rate-controlled encoding, so target = measured.

### Key Observations

1. **JPEG-AI Consistency:** Only codec with identical BPP across datasets (rate-controlled)
2. **Traditional Codec Variation:** JPEG/JPEG2000 show 1.5-2× higher BPP on RESISC45 (larger images, more detail)
3. **Neural Codec Efficiency:** MS-ILLM achieves lowest BPP at all quality levels
4. **Quality Parameters ≠ Bitrate:** Same q-level produces different BPP depending on content

---

## Key Results (EuroSAT - Complete)

### Baseline Accuracies
- **ResNet-18:** 97.65% test accuracy
- **ViT-S/16:** 99.73% test accuracy
- **Δ:** ViT outperforms by +2.08%

### Overall Codec Performance (ResNet-18)

**At Low Bitrate (q1, ~0.1-0.13 BPP):**
| Rank | Codec | Accuracy | BPP | Δ from Baseline |
|------|-------|----------|-----|-----------------|
| 1 | JPEG-AI | **77.73%** | 0.13 | -19.92% |
| 2 | MS-ILLM | 71.69% | 0.04 | -25.96% |
| 3 | JPEG | 46.99% | 0.77 | -50.66% |
| 4 | Cheng2020 | 34.30% | 0.13 | -63.35% |
| 5 | JPEG2000 | 13.54% | 0.10 | -84.11% |

**At High Bitrate (q6):**
| Rank | Codec | Accuracy | BPP | Δ from Baseline |
|------|-------|----------|-----|-----------------|
| 1 | JPEG2000 | **98.86%** | 0.97 | **+1.21%** |
| 2 | MS-ILLM | 97.75% | 0.90 | -0.10% |
| 3 | JPEG | 97.31% | 3.89 | -0.34% |
| 4 | JPEG-AI | 88.12% | 1.50 | -9.53% |
| 5 | Cheng2020 | 82.47% | 0.69 | -15.18% |

### Complete Accuracy Table (ResNet-18)

| Codec | q1 | q2 | q3 | q4 | q5 | q6 |
|-------|----|----|----|----|----|----|
| **JPEG** | 47.0% (0.77) | 67.1% (1.02) | 84.0% (1.34) | 91.6% (1.78) | 96.6% (2.67) | 97.3% (3.60) |
| **JPEG2000** | 13.5% (0.10) | 17.3% (0.15) | 41.8% (0.25) | 85.8% (0.45) | 98.8% (0.70) | 98.9% (0.97) |
| **Cheng2020** | 34.3% (0.13) | 40.4% (0.18) | 49.6% (0.26) | 63.7% (0.39) | 74.8% (0.53) | 82.5% (0.72) |
| **MS-ILLM** | 71.7% (0.04) | 81.2% (0.07) | 90.1% (0.14) | 95.4% (0.28) | 97.0% (0.56) | 97.8% (0.90) |
| **JPEG-AI** | 77.7% (0.13) | 83.0% (0.30) | 85.5% (0.47) | 86.7% (0.72) | 87.8% (1.07) | 88.1% (1.50) |

*(Numbers in parentheses = BPP)*

### Class-Specific Vulnerabilities (EuroSAT)

**Most Robust Classes:**
- **SeaLake:** >98% accuracy across all codecs/qualities (uniform texture, distinct color)
- **Forest:** 99.8% on JPEG/JPEG2000/MS-ILLM, but **0% on Cheng2020 q1-q5** (MSE destroys texture)

**Most Vulnerable Classes:**
- **Residential:** 30-80% accuracy drops at low bitrates (complex building patterns)
- **PermanentCrop:** Sensitive to artifact-induced blur (orchard row patterns destroyed)
- **Forest on MSE codecs:** Complete classification failure (explained below)

**Codec-Dependent Vulnerability:**
- **Cheng2020 (MSE-optimized):** Forest class catastrophic failure (0% q1-q5, 0.4% q6)
  - Explanation: MSE loss destroys homogeneous texture patterns distinguishing forest from vegetation
  - Dominant confusion: Forest → SeaLake (3,077 cases, smooth dark regions misclassified as water)
- **MS-ILLM (GAN-optimized):** Forest maintains 89.2% accuracy even at lowest quality
  - Explanation: GAN synthesizes plausible textures rather than smoothing
  - Preserves texture distributions essential for vegetation classification

### ViT vs ResNet Robustness

**Finding:** ViT-S/16 achieves higher baseline accuracy (+2.08%) but shows lower compression robustness than ResNet-18.

**Average Accuracy Drop Comparison:**
- **ViT-S/16:** Higher sensitivity to compression artifacts (larger drops)
- **ResNet-18:** More robust (smaller drops, especially at low bitrates)

**Interpretation:**
1. **Higher capacity ≠ Higher robustness:** ViT has 22M parameters vs ResNet's 11.2M
2. **Architectural differences:**
   - ViT: Patch-based (16×16), global self-attention → artifacts at patch boundaries disrupt
   - ResNet: Local convolutions (3×3) → more robust to localized artifacts
3. **Practical implication:** For compressed satellite imagery pipelines, CNNs preferred over Transformers

---

## Critical Findings for Thesis

### Finding 1: Loss Function Matters More Than Architecture

> **"GAN-based neural compression (MS-ILLM) achieves near-baseline classification accuracy (97.75%) at only 0.9 BPP, while MSE-optimized neural compression (Cheng2020) reaches only 82.47% even at its highest quality level (0.72 BPP). This 15-percentage-point gap demonstrates that perceptual loss functions preserve classification-relevant features better than pixel-wise reconstruction loss."**

**Supporting Evidence:**
- MS-ILLM: 97.75% at 0.9 BPP (Δ -0.10%)
- Cheng2020: 82.47% at 0.72 BPP (Δ -15.18%)
- Forest class: MS-ILLM 89.2%, Cheng2020 0.0% at q1

### Finding 2: JPEG-AI Ceiling Problem

> **"Despite being the ISO/ITU standardized neural codec, JPEG-AI exhibits a persistent accuracy ceiling of 88.12% even at 1.5 BPP—nearly 10 percentage points below the baseline. This is attributable to the mandatory `tools_off.json` configuration required for 64×64 images, which disables enhancement filters and resolution variance scaling. JPEG-AI's internal architecture expects larger input images and performs 3× upscaling to 192×192 before compression, introducing interpolation artifacts that persist through the compression-decompression cycle."**

**RESISC45 Note:** Native 256×256 resolution eliminates this problem (evaluation pending).

### Finding 3: Traditional Codecs Show Sharp Quality Thresholds

> **"JPEG2000 demonstrates a dramatic quality threshold: accuracy jumps from 41.79% at q3 (0.25 BPP) to 85.78% at q4 (0.45 BPP), and exceeds baseline at q5 (98.86% at 0.97 BPP). This suggests that JPEG2000 artifacts below a certain bitrate catastrophically destroy classification-relevant texture features, but above this threshold, the wavelet-based compression acts as mild denoising that slightly improves classification."**

**Exceeds Baseline:** JPEG2000 achieves +1.21% over baseline at q5-q6 (unexpected finding requiring discussion).

### Finding 4: MSE-Optimized Codecs Destroy Semantic Texture

> **"The 'Forest' class shows complete classification failure (0% accuracy) with MSE-optimized neural compression (Cheng2020) at quality levels q1-q5. The dominant misclassification pattern is Forest → SeaLake (3,077 cases, 19% of all misclassifications), suggesting that MSE-based smoothing produces uniform dark regions that the classifier interprets as water bodies. GAN-based compression (MS-ILLM) maintains 89.17% Forest accuracy even at its lowest quality level."**

**Implications:** For forestry/vegetation monitoring applications, GAN-based codecs strongly preferred.

### Finding 5: Bitrate Efficiency for Machine Vision

> **"MS-ILLM achieves the best bitrate efficiency for machine vision tasks: 97.75% accuracy at 0.9 BPP represents only 0.10% accuracy loss compared to uncompressed images. In contrast, JPEG requires 3.6 BPP to reach similar accuracy (97.31%), making MS-ILLM approximately 4× more efficient for satellite image classification pipelines."**

### Finding 6: Low-Bitrate Champion (Extreme Bandwidth)

> **"For extreme bandwidth constraints (<0.15 BPP), JPEG-AI outperforms all other codecs with 77.73% accuracy at 0.13 BPP. This makes JPEG-AI the preferred choice for satellite downlink scenarios where bandwidth is severely limited, despite its ceiling problem at higher bitrates."**

### Finding 7: CNN vs Transformer Robustness Trade-off

> **"Although Vision Transformers achieve higher baseline accuracy on uncompressed satellite images, they exhibit lower robustness to compression artifacts than CNNs. This has practical implications for architecture selection in compression-based remote sensing pipelines where models must handle variable-quality inputs."**

---

## Train-Once-Evaluate-Many Methodology

**Principle:** Classifiers are trained ONCE on original (high-quality) images and evaluated on all compressed versions WITHOUT retraining.

**Justification:**
1. **Realistic Deployment:** Operational systems train on archived high-quality data, then process incoming compressed imagery
2. **Codec Effect Isolation:** Training separate models per codec would conflate compression artifacts with model adaptation
3. **Literature Standard:** Dodge & Karam (2016), Paul et al. (2022), Varga et al. (2024) all use train-once methodology

**Alternative NOT Pursued:** Training codec-specific models (e.g., ResNet-on-JPEG, ResNet-on-Cheng2020) addresses a different research question: "Can models adapt to specific artifacts during training?" While scientifically interesting, this is less relevant for practical deployment where pre-trained models must handle diverse compression methods.

---

## Key Results (RESISC45 - Complete)

### Baseline Accuracies
- **ResNet-18:** 98.44% test accuracy
- **ViT-S/16:** 99.60% test accuracy
- **Δ:** ViT outperforms by +1.16%
- **Comparison to EuroSAT:** Both models achieve higher baseline on RESISC45 (+0.79% ResNet, +1.87% ViT)

### Overall Codec Performance at Quality Level q6

| Codec | BPP (q6) | ResNet-18 | ViT-S/16 | Δ from Baseline |
|-------|----------|-----------|----------|-----------------|
| **JPEG** | 2.230 | 98.10% | 99.57% | -0.03% (ViT) |
| **JPEG2000** | 2.386 | 97.88% | 99.52% | -0.08% (ViT) |
| **Cheng2020** | 0.982 | 97.75% | 99.45% | -0.15% (ViT) |
| **MS-ILLM** | 0.806 | 98.35% | 99.53% | -0.07% (ViT) |
| **JPEG-AI** | 1.500 | 98.48% | 99.49% | -0.11% (ViT) |

**Key Finding:** All codecs achieve within 0.7% of baseline at q6, demonstrating excellent compression robustness on higher-resolution images.

### Complete Accuracy Table (ViT-S/16 - RESISC45)

| Codec | q1 | q2 | q3 | q4 | q5 | q6 |
|-------|----|----|----|----|----|----|
| **JPEG** | N/A | N/A | N/A | N/A | N/A | 99.57% (2.23) |
| **JPEG2000** | N/A | N/A | N/A | N/A | N/A | 99.52% (2.39) |
| **Cheng2020** | N/A | N/A | 98.18% (0.30) | 98.83% (0.49) | 99.31% (0.71) | 99.45% (0.98) |
| **MS-ILLM** | 95.94% (0.04) | 96.73% (0.06) | 98.05% (0.12) | 98.75% (0.22) | 99.31% (0.43) | 99.53% (0.81) |
| **JPEG-AI** | 94.25% (0.12) | 98.45% (0.25) | 99.29% (0.50) | 99.42% (0.75) | 99.45% (1.00) | 99.49% (1.50) |

*(Numbers in parentheses = BPP)*

### Class-Specific Vulnerabilities (RESISC45)

**Most Robust Classes (JPEG-AI, ViT-S/16):**
- **Forest:** 100% at ALL quality levels (q1-q6)
- **Lake:** 99-100% across all qualities
- **Beach:** 99-100% across all qualities

**Most Vulnerable Classes:**
- **circular_farmland:** 90% at q1 → 100% at q6 (10% error rate at lowest quality)
- **dense_residential:** 93% at q1 → 99% at q6 (7% error rate at lowest quality)
- **medium_residential:** Similar pattern to dense_residential

**Pattern:** Visually similar classes (circular vs rectangular farmland, dense vs medium residential) show confusion at low bitrates, but all classes recover to >95% accuracy at q3 (0.50 BPP) and above.

### Misclassification Analysis (RESISC45)

**Total Misclassifications at q1 (Lowest Quality):**
- **ResNet-18:** 
  - JPEG-AI: 677 errors (91.19% acc)
  - MS-ILLM: 368 errors (95.21% acc)
  - Cheng2020: 1,326 errors (82.77% acc)
- **ViT-S/16:**
  - JPEG-AI: 443 errors (94.25% acc)
  - MS-ILLM: 313 errors (95.94% acc)
  - Cheng2020: 692 errors (91.01% acc)

**Total Misclassifications at q6 (Highest Quality):**
- **ResNet-18:**
  - JPEG-AI: 117 errors (98.48% acc)
  - MS-ILLM: 120 errors (98.43% acc)
  - Cheng2020: 137 errors (98.21% acc)
- **ViT-S/16:**
  - JPEG-AI: 39 errors (99.49% acc)
  - MS-ILLM: 39 errors (99.49% acc)
  - Cheng2020: 41 errors (99.47% acc)

**Key Finding:** At q6, all neural codecs achieve nearly identical performance with ViT-S/16 (39-41 errors out of 7,700 images).

### Cross-Dataset Comparison: EuroSAT vs RESISC45

**JPEG-AI Performance (ViT-S/16):**

| Quality | EuroSAT | RESISC45 | Δ (RESISC45 - EuroSAT) |
|---------|---------|----------|------------------------|
| Baseline | 98.42% | 99.60% | +1.18% |
| q1 (0.12-0.13 BPP) | 77.01% | 94.25% | **+17.24%** |
| q3 (0.50 BPP) | 95.51% | 99.29% | +3.78% |
| q6 (1.50 BPP) | 98.02% | 99.49% | +1.47% |

**Key Finding:** RESISC45 demonstrates significantly higher compression robustness, especially at low bitrates (+17.24% at q1). This is attributable to:
1. **Larger image size:** 256×256 provides more redundant information
2. **Higher baseline accuracy:** More distinguishing features in original images
3. **Better JPEG-AI compatibility:** Native resolution eliminates upscaling artifacts

### Critical Finding: Resolution Matters

> **"RESISC45's 256×256 resolution demonstrates 17.24 percentage points higher accuracy than EuroSAT at lowest quality (q1, 0.12 BPP). This massive difference suggests that compression robustness scales non-linearly with image size, and that EuroSAT's 64×64 resolution represents a worst-case scenario for neural compression."**

**Implications:**
- Operational satellite systems (typically >256×256) will show better compression robustness than EuroSAT results suggest
- JPEG-AI's ceiling problem on EuroSAT is resolution-specific, not fundamental to the codec
- Future research should prioritize realistic resolutions over benchmarking on small images

---

## Technical Implementation Details

### Quality Metrics Implementation

**PSNR (Peak Signal-to-Noise Ratio):**
- Formula: 10 × log₁₀(MAX²/MSE)
- Range: 20-50 dB typical
- Higher = better
- Issue: Poor correlation with perceptual quality

**SSIM (Structural Similarity Index):**
- Range: 0-1 (1 = identical)
- Considers luminance, contrast, structure
- Implementation fix for 64×64 images: win_size=7 (default 11 too large)

**MS-SSIM (Multi-Scale SSIM):**
- Multiple scales: 1×, 0.5×, 0.25×, 0.125×
- Implementation fix for 64×64: Limited to 2 scales (1×, 0.5×) due to size constraints
- Scales <7×7 skipped (SSIM window requirement)

**LPIPS (Learned Perceptual Image Patch Similarity):**
- AlexNet-based perceptual metric
- Range: 0-1 (lower = more similar)
- Best correlation with human perception
- Used for GAN-based codec evaluation

### Compression Pipeline Architecture

**Batch Processing:**
- Handles entire datasets with progress tracking (tqdm)
- Directory structure preserved (maintains class organization)
- Per-image metadata: original_size, compressed_size, BPP, compression_ratio

**Error Handling:**
- Individual image failures don't crash pipeline
- Try-except wraps for each metric calculation
- Partial results better than complete failure

**Output Structure:**
```
data/compressed/
├── jpeg_q10/ ... jpeg_q95/
├── jpeg2000_q1/ ... jpeg2000_q6/
├── cheng2020_q1/ ... cheng2020_q6/
├── msillm_q1/ ... msillm_q6/
└── jpegai_q1/ ... jpegai_q6/
```

### JPEG-AI Setup Challenges (RunPod)

**Requirements:**
- Ubuntu 20.04/22.04
- Git LFS (models stored as LFS pointers)
- Python 3.10
- Build tools (g++, make)
- NumPy <2.0 (compatibility)

**Critical Steps:**
1. `git lfs pull` to download actual models (~60MB .pth files)
2. Compile C++ extensions (ans.so, ec_direct.so) with pybind11
3. Install PyTorch CPU version (2.1.2+cpu)
4. Verify models loaded (check file sizes, not just existence)

**Common Issues Resolved:**
- Git LFS pointers instead of actual models
- C++ extensions not compiled
- NumPy 2.x incompatibility
- JPEG input not supported (requires PNG conversion)

---

## File Structure

```
MA-Neural_Compression_Satellite_Images/
├── data/
│   ├── EuroSAT_RGB/          # 27,000 images, 10 classes
│   ├── RESISC45_Subset/      # 7,700 images, 11 classes (NEW)
│   ├── splits/               # JSON split indices
│   └── compressed/           # All compressed versions
├── models/
│   ├── eurosat_resnet18.pth      # EuroSAT ResNet baseline
│   ├── eurosat_vit_s16.pth       # EuroSAT ViT baseline
│   ├── resisc45_resnet18.pth     # RESISC45 ResNet (NEW)
│   └── resisc45_vit_s16.pth      # RESISC45 ViT (NEW)
├── results/
│   ├── eurosat/              # EuroSAT evaluation results
│   ├── resisc45/             # RESISC45 evaluation results (PENDING)
│   ├── figures/              # Publication-ready plots
│   └── tables/               # LaTeX tables
├── notebooks/
│   ├── eurosat_training.ipynb
│   ├── resisc45_training.ipynb
│   └── compression_colab_*.ipynb
└── WORK_PROTOCOL.md          # This file (consolidated)
```

---

## Remaining Work

### Compression (In Progress)
- ⏳ RESISC45: Cheng2020 compression (1,155 images × 6 qualities)
- ⏳ RESISC45: MS-ILLM compression (7,700 images × 6 qualities) 
- ⏳ RESISC45: JPEG-AI compression (1,155 images × 6 qualities)

### Evaluation (Pending Compression)
- ⏳ RESISC45: ResNet-18 evaluation on all compressed versions
- ⏳ RESISC45: ViT-S/16 evaluation on all compressed versions
- ⏳ Cross-dataset comparison analysis
- ⏳ Class-specific vulnerability correlation (Spearman rank)

### Analysis
- ⏳ Rate-accuracy curves for RESISC45
- ⏳ Cross-dataset generalization assessment
- ⏳ Resolution-dependent codec behavior analysis
- ⏳ BD-Rate calculations (codec efficiency comparison)

### Visualization
- ⏳ RESISC45 rate-accuracy plots
- ⏳ Cross-dataset comparison figures
- ⏳ Per-class vulnerability heatmaps (both datasets)
- ⏳ Example compressed images (visual quality assessment)

### Documentation
- ⏳ Results chapter (Chapter 05) for both datasets
- ⏳ Discussion chapter (Chapter 06) updates
- ⏳ Introduction (Chapter 01) mention of second dataset
- ⏳ Conclusion (Chapter 07) generalization findings

---

## Key References (Cited in Thesis)

### Compression Methods
- **JPEG:** Wallace (1991) - The JPEG still picture compression standard
- **JPEG2000:** Taubman & Marcellin (2002) - JPEG2000: Image Compression Fundamentals
- **Cheng2020:** Cheng et al. (2020) - Learned image compression with discretized Gaussian mixture
- **MS-ILLM:** Muckley et al. (2023) - Improving statistical fidelity for neural image compression
- **JPEG-AI:** ISO/IEC 6048-1:2025 - Neural image coding

### Datasets
- **EuroSAT:** Helber et al. (2019) - EuroSAT: A novel dataset for land use and land cover classification
- **RESISC45:** Cheng et al. (2017) - Remote sensing image scene classification

### Methodology
- **Train-Once:** Dodge & Karam (2016), Paul et al. (2022), Varga et al. (2024)
- **Miscompressions:** Hofer & Böhme (2024) - On the Problem of Semantic Errors
- **ViT Robustness:** Bhojanapalli et al. (2021), Naseer et al. (2021)

### Related Work
- **Satellite Compression:** de Oliveira et al. (2022), Gomes et al. (2025)
- **Task-Aware Compression:** Lu et al. (2024), Li et al. (2024)
- **Spectral Bias:** Lieberman et al. (2023)

---

## Notes for Thesis Writing

### Chapter 02 (Background)
-  Rate-distortion theory (Shannon's theorem)
-  Traditional vs neural compression architectures
-  CNN vs Transformer architectures
-  Quality metrics (PSNR, SSIM, BD-Rate)
-  Miscompressions taxonomy (Hofer & Böhme)
-  NO CHANGES NEEDED (dataset-agnostic)

### Chapter 03 (Related Work)
-  Compression effects on machine learning (task-aware)
-  Satellite image compression (neural codecs for RS)
-  Semantic integrity (miscompressions)
-  ViT robustness studies
-  NO CHANGES NEEDED (covers both datasets)

### Chapter 04 (Methodology)
-  Both datasets fully described
-  Train-once methodology justified
-  Codec selection rationale clear
-  Training configs for both datasets
-  Baseline accuracies stated
-  Cross-dataset metrics defined
-  COMPLETE AND APPROVED

### Chapter 05 (Results) - TO BE WRITTEN
- Present EuroSAT complete results
- Add RESISC45 results when compression finishes
- Rate-accuracy curves for both datasets
- Class-specific analysis comparison
- BD-Rate codec comparison
- Cross-dataset generalization assessment

### Chapter 06 (Discussion) - TO BE UPDATED
- Interpret findings in light of RQs
- MSE vs GAN loss function implications
- JPEG-AI ceiling problem (EuroSAT) vs native resolution (RESISC45)
- Forest class failure mechanism (spectral bias + miscompressions)
- CNN vs ViT robustness trade-off
- JPEG2000 exceeds baseline phenomenon
- Cross-dataset generalization of findings
- Limitations: Both datasets JPEG-encoded sources

### Chapter 07 (Conclusion) - TO BE UPDATED
- Summarize key findings (both datasets)
- Practical recommendations by use case:
  - Extreme bandwidth (<0.15 BPP): JPEG-AI
  - Moderate bandwidth (0.5-1.0 BPP): MS-ILLM
  - High quality (>1.0 BPP): JPEG2000 or JPEG
  - Forestry applications: Avoid MSE codecs
- Future work: Additional datasets, fine-tuning on compressed, real-time compression

---

## Decision Log

### Why Two Datasets?
**Decision:** Add RESISC45 as second dataset for cross-validation

**Rationale:**
1. **Generalizability:** EuroSAT's 64×64 is unusually small; RESISC45's 256×256 more realistic
2. **Resolution dependence:** Test if findings hold at different scales
3. **JPEG-AI validation:** Native 256×256 eliminates upscaling workaround
4. **Thesis scope:** 11-class subset manageable within time constraints
5. **External validity:** Different source (Google Earth vs Sentinel-2)

**Trade-off:** More computational work, but significantly stronger thesis.

### Why Not Train Separate Models Per Codec?
**Decision:** Train once on originals, evaluate on all compressed versions

**Rationale:**
1. **Realistic deployment:** Pre-trained models must handle variable compression
2. **Fair comparison:** Isolates codec effect vs model adaptation
3. **Literature standard:** Dodge (2016), Paul (2022), Varga (2024) all use this approach
4. **Different RQ:** Compression-aware training is separate research question

**Alternative considered:** Train on compressed images
**Why not:** Would require 5 codecs × 6 qualities × 2 architectures × 2 datasets = 120 training runs. Infeasible for master's thesis.

### Why These Specific Codecs?
**Decision:** JPEG, JPEG2000, Cheng2020, MS-ILLM, JPEG-AI

**Rationale:**
1. **JPEG:** Ubiquitous baseline, DCT-based traditional
2. **JPEG2000:** Remote sensing standard, wavelet-based traditional
3. **Cheng2020:** Established MSE-optimized neural (reproducible via CompressAI)
4. **MS-ILLM:** State-of-art GAN-based perceptual neural
5. **JPEG-AI:** First standardized neural codec (ISO/IEC)

**Why not SOTA (ELIC, LIC-TCM)?:** Reproducibility > performance. Cheng2020 has pretrained weights, 2800+ citations, clear optimization objective.

### Why ViT in Addition to ResNet?
**Decision:** Evaluate both CNN (ResNet-18) and Transformer (ViT-S/16)

**Rationale:**
1. **Architectural diversity:** Test if findings generalize across paradigms
2. **Literature gap:** ViT compression robustness understudied for satellite imagery
3. **Hypothesis:** Local convolutions might be more robust than global attention
4. **Thesis contribution:** First systematic CNN vs ViT comparison for compressed satellite images

**Finding:** Confirmed hypothesis - ResNet more robust despite lower baseline accuracy.

---

## Computational Resources Used

### Google Colab (Training)
- **GPU:** NVIDIA Tesla T4
- **Usage:** All model training (4 models × 2 datasets = 8 training runs)
- **Compute units consumed:** ~100-150 units (purchased after free tier depleted)
- **Training times:**
  - EuroSAT ResNet: ~90 min (local Mac CPU, initial)
  - EuroSAT ViT: ~10-12 min (Colab T4)
  - RESISC45 ResNet: ~20-30 min (Colab T4)
  - RESISC45 ViT: ~15-20 min (Colab T4)

### RunPod (Compression)
- **Hardware:** Various pods with 16-48 vCPUs, some with GPUs (RTX 3090/4090)
- **Usage:** MS-ILLM and JPEG-AI compression
- **Total cost:** ~$50-70 (JPEG-AI especially expensive due to slow encode)
- **Compression times:**
  - MS-ILLM EuroSAT: ~4-5 hours (2× RTX 4090 GPU, q1-q6)
  - MS-ILLM RESISC45: ~6-8 hours (2× RTX 4090 GPU, q1-q6)
  - Cheng2020 EuroSAT/RESISC45: ~2-3 hours (RTX 3090 GPU)
  - JPEG-AI EuroSAT: ~24 hours (RTX 3090 GPU, sequential processing)
  - JPEG-AI RESISC45: ~12 hours (CPU-only with 8 parallel workers, 16 vCPUs)

**JPEG-AI Hardware Note:**
- EuroSAT (64×64): GPU mode with sequential processing (requires upscaling workaround)
- RESISC45 (256×256): CPU mode with 8 parallel workers (ProcessPoolExecutor)
  - GPU mode caused C++ extension build issues on some pods
  - CPU parallel processing was more stable and faster overall for larger images

### Local Mac (Development & Traditional Codecs)
- **Hardware:** 2020 MacBook Pro, 8GB RAM, Intel i5
- **Usage:** JPEG/JPEG2000 compression, script development, thesis writing
- **Limitation:** No CUDA (MPS not stable for training)

---

## Status Summary

** COMPLETE:**
- EuroSAT dataset acquisition & preprocessing
- EuroSAT baseline training (ResNet-18, ViT-S/16)
- EuroSAT compression (all 5 codecs, all quality levels)
- EuroSAT evaluation (both models, all compressed versions)
- EuroSAT analysis & visualization
- RESISC45 dataset acquisition & preprocessing
- RESISC45 baseline training (ResNet-18, ViT-S/16)
- **RESISC45 compression (all 5 codecs, all quality levels)**  NEW
- **RESISC45 evaluation (ResNet-18, ViT-S/16 on all codecs)**  NEW
- **RESISC45 misclassification analysis**  NEW
- **Master results consolidation (master_results.json, measured_bpp_all_codecs.json)**  NEW
- **Cross-dataset comparison figures**  NEW
- Methodology chapter (Chapter 04) finalized
- **Results chapter (Chapter 05) updated with RESISC45 section**  NEW

**⏳ IN PROGRESS:**
- Discussion chapter (Chapter 06) updates with cross-dataset findings
- Introduction (Chapter 01) RESISC45 mention
- Conclusion (Chapter 07) generalization findings

**📋 PENDING:**
- Final proofreading
- Abstract finalization
- Bibliography check

**🎯 DEADLINE:**
- Thesis submission: December 19, 2025
- Days remaining: 3 days

---

---

## December 16, 2025 Updates

### Completed Today
1. **RESISC45 Compression Complete:**
   - All 5 codecs fully compressed at all quality levels
   - Total: 150,150 compressed images
   - JPEG-AI: 46,200 images (all 7,700 test images × 6 qualities)

2. **RESISC45 Evaluation Complete:**
   - ResNet-18 and ViT-S/16 evaluated on all compressed versions
   - Results stored in `results/resisc45/` directory
   - Key finding: 17.24% higher accuracy at q1 compared to EuroSAT

3. **BPP Measurement Correction:**
   - Created `measured_bpp_all_codecs.json` with actual logged bitrates
   - Corrected RESISC45 BPP values (previously estimated)
   - Added BPP argumentation to thesis (chap04.tex)

4. **Master Results Consolidation:**
   - `master_results.json`: All evaluation results in single file
   - `measured_bpp_all_codecs.json`: All bitrate measurements
   - Enables reproducible plot generation

5. **Visualization Updates:**
   - `generate_plots_from_master.py`: Consolidated plot script using master results
   - Generated 6 new RESISC45 plots:
     - Rate-accuracy comparison
     - ViT vs ResNet comparison
     - JPEG-AI detailed performance
     - JPEG-AI class heatmap
     - Cross-dataset comparison
     - Compression examples
   - Created misclassification example figures for both datasets

6. **Thesis Updates:**
   - Chapter 04: Added RESISC45 dataset section with class examples figure
   - Chapter 05: Added complete RESISC45 results section
   - Chapter 05: Added misclassification example figures
   - Fixed Table 3 (codec overview) width overflow
   - Fixed Unicode characters (°, ×, Δ) → LaTeX math
   - Added missing bibliography entry (cheng2017remote)
   - Thesis compiles successfully

7. **Misclassification Analysis:**
   - `resisc45_miscompression_analysis.json`: Detailed breakdown
   - Identified most vulnerable classes (circular_farmland, dense_residential)
   - Analyzed confusion patterns at q1 vs q6

8. **Cheng2020 Misclassification Analysis Completed (q1-q4):**
   - Created `scripts/analysis/complete_cheng2020_eurosat.py` with auto-save functionality
   - Completed evaluation for q1-q4 that was previously missing
   - Results: q1: 5,338 misclassifications, q2: 4,858, q3: 4,074, q4: 2,990
   - Stored in `results/cheng2020_misclassifications_q1_q4.json`
   - Updated thesis appendix Table (app01.tex) with complete q1-q6 data
   - Removed outdated note about missing q1-q4 data

### Key Technical Achievements
- **Reproducibility:** All plots now generated from master results file
- **Documentation:** Complete BPP measurements for both datasets
- **Consistency:** Standardized analysis pipeline across datasets
- **Thesis Integration:** All new findings incorporated with figures

### Files Created Today
- `scripts/generate_plots_from_master.py`
- `scripts/analyze_resisc45_miscompressions.py`
- `scripts/generate_misclassification_figures.py`
- `scripts/create_master_results_v2.py`
- `results/master_results.json`
- `results/measured_bpp_all_codecs.json`
- `results/resisc45_miscompression_analysis.json`
- `docs/thesis_fhkufstein/img/resisc45_*.pdf` (7 new figures)
- `docs/thesis_fhkufstein/img/eurosat_misclassification_examples.pdf`

---

**END OF CONSOLIDATED WORK PROTOCOL**

*All experiments, findings, and technical details documented as of December 16, 2025. RESISC45 dataset fully completed and integrated into thesis. For the most current status, refer to Git commits and results/ directory.*
