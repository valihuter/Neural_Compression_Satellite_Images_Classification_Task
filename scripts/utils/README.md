# Scripts Utils

Gemeinsame Utility-Module für Scripts.

## model_utils.py

Enthält:
- `BaselineClassifier`: ResNet-18 Wrapper für EuroSAT Klassifikation
- `EuroSATDataset`: PyTorch Dataset für EuroSAT RGB

Verwendung:
```python
from scripts.utils.model_utils import BaselineClassifier

model = BaselineClassifier(num_classes=10, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```

Diese Module wurden aus dem WP-Ordner hierher verschoben, um die Abhängigkeiten von Prototyp-Code zu entfernen.
