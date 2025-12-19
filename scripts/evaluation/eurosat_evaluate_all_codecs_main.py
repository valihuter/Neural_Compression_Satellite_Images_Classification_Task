#!/usr/bin/env python3
"""
Evaluation Script - EuroSAT Dataset (All Codecs)

Dataset: EuroSAT RGB (64×64, 10 classes, 4,050 test images)
Models: ResNet-18, ViT-S/16
Hardware: Local Mac M1 (MPS) or RunPod GPU

Evaluates classification accuracy on compressed images for all codecs:
    - JPEG (q1-q6)
    - JPEG2000 (q1-q6)
    - Cheng2020 (q3-q6, q1-q2 too aggressive)
    - MS-ILLM (q1-q6)
    - JPEG-AI (q1-q6)

Usage:
    python eurosat_evaluate.py --codec jpeg --model resnet18
    python eurosat_evaluate.py --codec all --model all
    python eurosat_evaluate.py --reference  # Show reference results
=============================================================================
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm

# Configuration
DATASET = "EuroSAT"
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 10
BATCH_SIZE = 64

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "eurosat"
RESULTS_DIR = BASE_DIR / "results" / "json"

# Compressed data paths
EUROSAT_BASE = DATA_DIR / "eurosat"
COMPRESSED_PATHS = {
    "original": EUROSAT_BASE / "raw",
    "jpeg": EUROSAT_BASE / "comp" / "jpeg",
    "jpeg2000": EUROSAT_BASE / "comp" / "jpeg2000",
    "cheng2020": EUROSAT_BASE / "comp" / "cheng2020",
    "msillm": EUROSAT_BASE / "comp" / "msillm",
    "jpegai": EUROSAT_BASE / "comp" / "jpegai",
}

MODEL_PATHS = {
    "resnet18": MODELS_DIR / "resnet18_classifier.pth",
    "vit": MODELS_DIR / "vit_classifier.pth",
}

# Reference results (December 2025)
REFERENCE_RESULTS = {
    "resnet18": {
        "original": 97.65,
        "jpeg": {"q1": 68.10, "q2": 86.42, "q3": 94.15, "q4": 96.47, "q5": 97.41, "q6": 97.58},
        "jpeg2000": {"q1": 71.38, "q2": 83.06, "q3": 92.20, "q4": 95.85, "q5": 97.19, "q6": 97.51},
        "cheng2020": {"q3": 87.90, "q4": 93.75, "q5": 95.95, "q6": 97.01},
        "msillm": {"q1": 83.01, "q2": 89.73, "q3": 93.43, "q4": 95.68, "q5": 97.04, "q6": 97.75},
        "jpegai": {"q1": 77.73, "q2": 82.96, "q3": 85.51, "q4": 86.69, "q5": 87.75, "q6": 88.12},
    },
    "vit": {
        "original": 98.84,
        "jpeg": {"q1": 76.94, "q2": 92.05, "q3": 97.33, "q4": 98.47, "q5": 98.72, "q6": 98.77},
        "jpeg2000": {"q1": 81.26, "q2": 90.17, "q3": 96.25, "q4": 98.22, "q5": 98.69, "q6": 98.79},
        "cheng2020": {"q3": 94.69, "q4": 97.41, "q5": 98.32, "q6": 98.64},
        "msillm": {"q1": 91.36, "q2": 95.51, "q3": 97.43, "q4": 98.22, "q5": 98.57, "q6": 98.77},
        "jpegai": {"q1": 77.01, "q2": 84.69, "q3": 88.72, "q4": 90.35, "q5": 91.41, "q6": 91.90},
    }
}

def load_model(model_name: str, device: str):
    """Load classifier model."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        checkpoint_path = MODEL_PATHS["resnet18"]
    else:
        import timm
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
        checkpoint_path = MODEL_PATHS["vit"]

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    return model.eval().to(device)

def get_transforms():
    """Get image transforms."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def evaluate(model, dataloader, device):
    """Evaluate model accuracy."""
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for pred, label in zip(predicted, labels):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1

    accuracy = 100 * correct / total
    per_class = {CLASSES[i]: 100 * class_correct[i] / class_total[i] for i in range(NUM_CLASSES)}
    return accuracy, per_class

def print_reference():
    """Print reference results."""
    print("\nReference Results (December 2025):")
    for model_name in ["resnet18", "vit"]:
        print(f"\n{model_name.upper()}:")
        ref = REFERENCE_RESULTS[model_name]
        print(f"  Original: {ref['original']:.2f}%")
        for codec in ["jpeg", "jpeg2000", "cheng2020", "msillm", "jpegai"]:
            if codec in ref:
                vals = ref[codec]
                q1 = vals.get("q1", vals.get("q3"))
                q6 = vals["q6"]
                print(f"  {codec}: q1={q1:.2f}%, q6={q6:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT classification")
    parser.add_argument("--codec", choices=["original", "jpeg", "jpeg2000", "cheng2020", "msillm", "jpegai", "all"])
    parser.add_argument("--quality", choices=["q1", "q2", "q3", "q4", "q5", "q6", "all"], default="all")
    parser.add_argument("--model", choices=["resnet18", "vit", "all"], default="all")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--reference", action="store_true", help="Print reference results")
    args = parser.parse_args()

    if args.reference:
        print_reference()
        return

    if not args.codec:
        parser.error("--codec is required (or use --reference)")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    codecs = ["original", "jpeg", "jpeg2000", "cheng2020", "msillm", "jpegai"] if args.codec == "all" else [args.codec]
    qualities = ["q1", "q2", "q3", "q4", "q5", "q6"] if args.quality == "all" else [args.quality]
    models_to_eval = ["resnet18", "vit"] if args.model == "all" else [args.model]

    results = {"dataset": DATASET, "timestamp": datetime.now().isoformat(), "evaluations": []}
    transform = get_transforms()

    for model_name in models_to_eval:
        print(f"\n{'='*60}\nLoading {model_name}...")
        model = load_model(model_name, device)

        for codec in codecs:
            base_path = COMPRESSED_PATHS[codec]
            codec_qualities = ["original"] if codec == "original" else qualities

            for quality in codec_qualities:
                data_path = base_path if codec == "original" else base_path / quality
                if not data_path.exists():
                    print(f"  Skipped {codec}/{quality}: path not found")
                    continue

                print(f"\n{model_name} on {codec}/{quality}...")
                dataset = datasets.ImageFolder(data_path, transform=transform)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                accuracy, per_class = evaluate(model, dataloader, device)

                results["evaluations"].append({
                    "model": model_name, "codec": codec, "quality": quality,
                    "accuracy": round(accuracy, 2), "per_class": {k: round(v, 2) for k, v in per_class.items()}
                })
                print(f"  Accuracy: {accuracy:.2f}%")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
