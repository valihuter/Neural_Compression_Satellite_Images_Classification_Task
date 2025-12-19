# Naming Convention für Results

## Einheitliche Benennung

### Datasets
| JSON Key | Display (Thesis) | Ordner |
|----------|------------------|--------|
| `eurosat` | EuroSAT | `eurosat/` |
| `resisc45` | RESISC45 | `resisc45/` |

### Models
| JSON Key | Display (Thesis) | Ordner |
|----------|------------------|--------|
| `resnet18` | ResNet-18 | `resnet18/` |
| `vit` | ViT-S/16 | `vit/` |

**Hinweis:** `vit_s16`, `vit_small`, `vit_small_patch16_224` → alle vereinheitlichen zu `vit`

### Codecs
| JSON Key | Display (Thesis) | Ordner | Typ |
|----------|------------------|--------|-----|
| `jpeg` | JPEG | `jpeg/` | Traditional (DCT) |
| `jpeg2000` | JPEG2000 | `jpeg2000/` | Traditional (Wavelet) |
| `cheng2020` | Cheng2020-attn | `cheng2020/` | Neural (CNN) |
| `msillm` | MS-ILLM | `msillm/` | Neural (LLM) |
| `jpegai` | JPEG-AI | `jpegai/` | Neural (Standard) |

**Hinweis:** Es wurde nur `cheng2020-attn` (Attention-Variante) verwendet, nicht die Basis-Variante. In JSON und Ordnern einheitlich als `cheng2020` bezeichnet.

### Quality Levels
| JSON Key | Ordner | Bedeutung |
|----------|--------|-----------|
| `q1` | `q1/` | Niedrigste Qualität / höchste Kompression |
| `q2` | `q2/` | ... |
| `q3` | `q3/` | ... |
| `q4` | `q4/` | ... |
| `q5` | `q5/` | ... |
| `q6` | `q6/` | Höchste Qualität / niedrigste Kompression |

---

## Ordnerstruktur

### Data Ordner
```
data/
├── eurosat/
│   ├── raw/                    # Unkomprimierte Originaldaten
│   │   └── {class}/            # z.B. Forest/, River/, ...
│   └── comp/                   # Komprimierte Daten
│       ├── jpeg/
│       │   └── q{1-6}/
│       │       └── {class}/
│       ├── jpeg2000/
│       │   └── q{1-6}/
│       │       └── {class}/
│       ├── cheng2020/
│       │   └── q{1-6}/
│       │       └── {class}/
│       ├── msillm/
│       │   └── q{1-6}/
│       │       └── {class}/
│       └── jpegai/
│           └── q{1-6}/
│               └── {class}/
│
└── resisc45/
    ├── raw/
    │   └── {class}/
    └── comp/
        ├── jpeg/
        │   └── q{1-6}/
        │       └── {class}/
        └── ... (analog zu eurosat)
```

### Models Ordner
```
models/
├── eurosat/
│   ├── resnet18/
│   │   └── best_model.pth
│   └── vit/
│       └── best_model.pth
└── resisc45/
    ├── resnet18/
    │   └── best_model.pth
    └── vit/
        └── best_model.pth
```

### Results Ordner
```
results/
├── json/
│   ├── master_results.json
│   ├── bpp_measurements_all.json
│   ├── {dataset}_{model}_{codec}_evaluation.json
│   └── ...
├── csv/
├── class_analysis/
└── miscompression_analysis/
```

---

## Dateistruktur JSON
```
results/json/
├── master_results.json          # Konsolidierte Ergebnisse
├── bpp_measurements_all.json    # BPP-Werte
├── bd_rate_analysis.json        # BD-Rate Analyse
├── {dataset}_{model}_{codec}_evaluation.json
├── {dataset}_{model}_{codec}_class_evaluation.json
├── {dataset}_{model}_{codec}_misclassifications.json
└── {dataset}_miscompression_analysis.json
```

### master_results.json Struktur
```json
{
  "metadata": { ... },
  "bpp_data": { ... },
  "eurosat": {
    "resnet18": {
      "baseline": { "accuracy": ..., "class_accuracies": { ... } },
      "jpeg": { "quality_levels": { "q1": { "accuracy": ..., "class_accuracies": { ... } }, ... } },
      "jpeg2000": { ... },
      "cheng2020": { ... },
      "msillm": { ... },
      "jpegai": { ... }
    },
    "vit": { ... }
  },
  "resisc45": { ... }
}
```
