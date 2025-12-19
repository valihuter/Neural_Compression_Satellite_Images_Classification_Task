#!/usr/bin/env python3
"""
Evaluation Script - RESISC45 Subset Dataset (All Codecs)

Dataset: RESISC45 Subset (256×256, 11 classes, 7,700 images)
Models: ResNet-18, ViT-S/16
Hardware: Local Mac M1 (MPS) or RunPod GPU

Classes: airplane, airport, baseball_diamond, beach, bridge, commercial_area,
         dense_residential, freeway, golf_course, ground_track_field, harbor

Evaluates classification accuracy on compressed images for all codecs:
    - JPEG (q1-q6)
    - JPEG2000 (q1-q6)
    - Cheng2020 (q3-q6)
    - MS-ILLM (q1-q6)
    - JPEG-AI (q1-q6)

Usage:
    python resisc45_evaluate.py --codec jpeg --model resnet18
    python resisc45_evaluate.py --codec all --model all
    python resisc45_evaluate.py --reference  # Show reference results
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
DATASET = "RESISC45_SUBSET11"
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 11
BATCH_SIZE = 32

CLASSES = [
    "airplane", "airport", "baseball_diamond", "beach", "bridge",
    "commercial_area", "dense_residential", "freeway", "golf_course",
    "ground_track_field", "harbor"
]

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "resisc45"
RESULTS_DIR = BASE_DIR / "results" / "json"

# Compressed data paths
RESISC45_BASE = DATA_DIR / "resisc45"
COMPRESSED_PATHS = {
    "original": RESISC45_BASE / "raw",
    "jpeg": RESISC45_BASE / "comp" / "jpeg",
    "jpeg2000": RESISC45_BASE / "comp" / "jpeg2000",
    "cheng2020": RESISC45_BASE / "comp" / "cheng2020",
    "msillm": RESISC45_BASE / "comp" / "msillm",
    "jpegai": RESISC45_BASE / "comp" / "jpegai",
}

MODEL_PATHS = {
    "resnet18": MODELS_DIR / "resnet18_classifier.pth",
    "vit": MODELS_DIR / "vit_classifier.pth",
}

# Reference results (December 2025)
REFERENCE_RESULTS = {
    "resnet18": {
        "original": 96.00,
        "jpeg": {"q1": 58.05, "q2": 78.55, "q3": 91.20, "q4": 94.40, "q5": 95.50, "q6": 95.95},
        "jpeg2000": {"q1": 76.45, "q2": 85.20, "q3": 92.10, "q4": 94.65, "q5": 95.70, "q6": 96.00},
        "cheng2020": {"q3": 89.95, "q4": 93.45, "q5": 95.05, "q6": 95.70},
        "msillm": {"q1": 78.70, "q2": 86.40, "q3": 91.85, "q4": 94.25, "q5": 95.40, "q6": 95.80},
        "jpegai": {"q1": 93.35, "q2": 95.15, "q3": 95.45, "q4": 95.50, "q5": 95.55, "q6": 95.60},
    },
    "vit": {
        "original": 97.90,
        "jpeg": {"q1": 66.00, "q2": 85.40, "q3": 95.10, "q4": 97.25, "q5": 97.70, "q6": 97.90},
        "jpeg2000": {"q1": 84.70, "q2": 91.85, "q3": 96.10, "q4": 97.50, "q5": 97.85, "q6": 98.00},
        "cheng2020": {"q3": 95.00, "q4": 96.95, "q5": 97.55, "q6": 97.80},
        "msillm": {"q1": 84.50, "q2": 91.20, "q3": 95.60, "q4": 97.05, "q5": 97.60, "q6": 97.80},
        "jpegai": {"q1": 94.25, "q2": 98.45, "q3": 99.29, "q4": 99.42, "q5": 99.45, "q6": 97.75},
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
        print(f"WARNING: Model not found: {checkpoint_path}")
        print("Using randomly initialized weights (results will be incorrect)")

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
    print("\nReference Results - RESISC45 (December 2025):")
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
    
    print("\n" + "=" * 70)
    print("Key Finding: JPEG-AI on RESISC45 (256×256)")
    print("  ViT q1 (0.12 BPP): 94.25% accuracy")
    print("  vs EuroSAT (64×64) q1: 77.01% accuracy")
    print("  Δ = +17.24 pp (no upscaling workaround needed)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RESISC45 classification")
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

    results = {"dataset": DATASET, "timestamp": datetime.now().isoformat(), "classes": CLASSES, "evaluations": []}
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
