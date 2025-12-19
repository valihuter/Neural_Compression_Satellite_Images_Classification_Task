# Neural Compression for Satellite Images

Master's Thesis: Impact of Neural Image Compression on Satellite Image Classification

## Research Objective

This thesis provides a systematic evaluation of neural compression methods for satellite imagery, focusing on their impact on downstream classification tasks. The study compares traditional codecs (JPEG, JPEG2000) with state-of-the-art neural compression methods (Cheng2020, MS-ILLM, JPEG-AI) using the EuroSAT dataset.

## Research Questions

1. How does neural compression affect land-use classification accuracy compared to traditional codecs?
2. At which bitrates do neural methods outperform traditional compression?
3. Which image classes are most sensitive to compression artifacts?

## Dataset

**EuroSAT RGB** (Helber et al., 2019)
- 27,000 Sentinel-2 satellite images
- 64x64 pixels, 3 channels (RGB)
- 10 land-use classes
- Split: 70% train / 15% validation / 15% test

## Compression Methods

| Codec | Type | Reference |
|-------|------|-----------|
| JPEG | Traditional (DCT) | ISO/IEC 10918 |
| JPEG2000 | Traditional (Wavelet) | ISO/IEC 15444 |
| Cheng2020-attn | Neural (Attention) | Cheng et al., CVPR 2020 |
| MS-ILLM | Neural (GAN) | Muckley et al., ICML 2023 |
| JPEG-AI | Neural (ISO Standard) | ISO/IEC 21122 |

## Project Structure

```
├── scripts/                 # Compression and evaluation scripts
│   ├── compress_jpeg2000.py
│   ├── compress_msillm.py
│   ├── compress_cheng2020_finetuned.py
│   └── finetune_cheng2020_eurosat.py
├── cloud-gpu/               # RunPod GPU scripts (JPEG-AI)
├── WP2-Data-Preparation/    # Data loading and preprocessing
├── WP3-Compression-Implementation/  # Codec implementations
├── WP4-Downstream-Evaluation/       # Classification evaluation
├── background/              # Literature and references
├── data/                    # Datasets (not in repository)
├── results/                 # Evaluation results
└── models/                  # Trained model weights
```

## Baseline Results

ResNet-18 classifier trained on uncompressed EuroSAT:
- Test Accuracy: 97.65%
- Training: 20 epochs, Adam optimizer, lr=0.001

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0
- CompressAI >= 1.2.8
- Pillow, scikit-image
- glymur (JPEG2000)

## Usage

### Compress Dataset

```bash
# JPEG2000
python scripts/compress_jpeg2000.py

# MS-ILLM (requires GPU)
python scripts/compress_msillm.py

# JPEG-AI (RunPod)
python cloud-gpu/compress_eurosat_batch.py
```

### Evaluate Compression

```bash
python WP4-Downstream-Evaluation/evaluate_compressed.py
```

## References

- Helber, P., et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS.
- Cheng, Z., et al. (2020). Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. CVPR.
- Muckley, M., et al. (2023). Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models. ICML.

## Author

Vali Huter  
Master's Thesis, 2025
