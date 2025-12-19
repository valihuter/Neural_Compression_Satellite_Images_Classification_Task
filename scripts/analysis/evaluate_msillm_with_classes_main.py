#!/usr/bin/env python3
"""
MS-ILLM EuroSAT Test Set Evaluation with Per-Class Results
For consistency with other codecs.
"""
import sys
import json
from pathlib import Path
from collections import defaultdict
import torch
from torchvision import transforms
from PIL import Image
import timm

# Paths
PROJECT_ROOT = Path("/Users/vali/MA-Neural_Compression_Satellite_Images")
COMPRESSED_BASE = PROJECT_ROOT / "data" / "EUROSAT" / "comp" / "msillm"
SPLIT_FILE = PROJECT_ROOT / "data" / "EUROSAT" / "uncomp" / "splits" / "rgb_test.json"
RESNET_MODEL_PATH = PROJECT_ROOT / "models" / "EUROSAT" / "baseline_classifier.pth"
VIT_MODEL_PATH = PROJECT_ROOT / "models" / "EUROSAT" / "vit_classifier.pth"
OUTPUT_FILE = PROJECT_ROOT / "results" / "json" / "eurosat_msillm_class_evaluation.json"

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
           'River', 'SeaLake']

QUALITY_LEVELS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load models
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from utils.model_utils import BaselineClassifier

print("Loading ResNet-18...")
resnet = BaselineClassifier(num_classes=10, pretrained=False)
checkpoint = torch.load(RESNET_MODEL_PATH, map_location=device, weights_only=False)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet.to(device).eval()

print("Loading ViT-S/16...")
vit = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
checkpoint = torch.load(VIT_MODEL_PATH, map_location=device, weights_only=False)
vit.load_state_dict(checkpoint['model_state_dict'])
vit.to(device).eval()

# Load test split
with open(SPLIT_FILE, 'r') as f:
    test_files = json.load(f)

# Build lookup: filename -> (class_name, label)
test_file_map = {}
for img_path in test_files:
    class_name = img_path.split('/')[0]
    filename = img_path.split('/')[-1]
    label = CLASSES.index(class_name)
    # Store without extension for matching
    basename = filename.replace('.jpg', '').replace('.png', '')
    test_file_map[basename] = (class_name, label)

print(f"Loaded {len(test_file_map)} test images\n")

# Transforms
transform_resnet = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

results = {
    'codec': 'MS-ILLM',
    'dataset': 'EuroSAT',
    'split': 'test',
    'num_test_images': len(test_file_map),
    'models': {}
}

for model_name, model, transform in [
    ('ResNet-18', resnet, transform_resnet),
    ('ViT-S/16', vit, transform_vit)
]:
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    model_results = {}
    
    for q in QUALITY_LEVELS:
        print(f"Processing {q}...")
        compressed_dir = COMPRESSED_BASE / f"msillm_{q}"
        
        if not compressed_dir.exists():
            print(f"  Warning: {compressed_dir} not found, skipping")
            continue
        
        # Per-class counters
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        total_correct = 0
        total_processed = 0
        
        for class_name in CLASSES:
            class_dir = compressed_dir / class_name
            if not class_dir.exists():
                print(f"  Warning: {class_dir} not found")
                continue
            
            for img_path in class_dir.glob("*.png"):
                filename = img_path.name
                basename = filename.replace('.png', '')
                
                # Check if this image is in test set
                if basename not in test_file_map:
                    continue
                
                expected_class, true_label = test_file_map[basename]
                
                # Verify class consistency
                if expected_class != class_name:
                    print(f"  Warning: Class mismatch for {filename}")
                    continue
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    x = transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred = model(x).argmax(dim=1).item()
                    
                    class_total[class_name] += 1
                    total_processed += 1
                    
                    if pred == true_label:
                        class_correct[class_name] += 1
                        total_correct += 1
                        
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    continue
        
        # Calculate accuracies
        overall_accuracy = (total_correct / total_processed * 100) if total_processed > 0 else 0.0
        
        class_accuracies = {}
        for class_name in CLASSES:
            if class_total[class_name] > 0:
                acc = class_correct[class_name] / class_total[class_name] * 100
                class_accuracies[class_name] = {
                    'accuracy': round(acc, 2),
                    'correct': class_correct[class_name],
                    'total': class_total[class_name]
                }
            else:
                class_accuracies[class_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }
        
        model_results[q] = {
            'overall_accuracy': round(overall_accuracy, 2),
            'total_correct': total_correct,
            'total_processed': total_processed,
            'class_results': class_accuracies
        }
        
        print(f"  {q}: {overall_accuracy:.2f}% ({total_correct}/{total_processed})")
    
    results['models'][model_name] = model_results

# Save results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to: {OUTPUT_FILE}")
print(f"{'='*60}\n")

# Print summary
print("\n=== SUMMARY ===")
for model_name in results['models']:
    print(f"\n{model_name}:")
    for q in QUALITY_LEVELS:
        if q in results['models'][model_name]:
            acc = results['models'][model_name][q]['overall_accuracy']
            print(f"  {q}: {acc:.2f}%")
