#!/usr/bin/env python3
"""
Complete Cheng2020 EuroSAT misclassification analysis for q1-q4.
Quick script to fill the gap in the misclassification analysis.
"""
import sys
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import timm

# Paths
PROJECT_ROOT = Path("/Users/vali/MA-Neural_Compression_Satellite_Images")
ORIGINAL_DATA = PROJECT_ROOT / "data" / "EUROSAT" / "UNCOMP" / "EuroSAT_RGB"
COMPRESSED_BASE = PROJECT_ROOT / "data" / "EUROSAT" / "COMP" / "cheng2020-attn"
SPLIT_FILE = PROJECT_ROOT / "data" / "EUROSAT" / "UNCOMP" / "splits" / "rgb_test.json"
MODEL_PATH = PROJECT_ROOT / "models" / "EUROSAT" / "baseline_classifier.pth"
VIT_MODEL_PATH = PROJECT_ROOT / "models" / "EUROSAT" / "vit_classifier.pth"

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
           'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
           'River', 'SeaLake']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load models
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from utils.model_utils import BaselineClassifier

print("Loading ResNet-18...")
resnet = BaselineClassifier(num_classes=10, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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

file_to_label = {}
test_file_set = set()
for img_path in test_files:
    class_name = img_path.split('/')[0]
    filename = img_path.split('/')[-1]
    label = CLASSES.index(class_name)
    file_to_label[filename] = label
    # Store without extension for matching
    basename = filename.replace('.jpg', '').replace('.png', '')
    test_file_set.add(basename)

print(f"Loaded {len(file_to_label)} test images\n")

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

# Output file for intermediate results
OUTPUT_FILE = PROJECT_ROOT / "results" / "cheng2020_misclassifications_q1_q4.json"

# Load existing results if available
results = {}
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, 'r') as f:
        results = json.load(f)
    print(f"Loaded existing results: {list(results.keys())}\n")

for q in ['q1', 'q2', 'q3', 'q4']:
    # Skip if already processed
    if q in results:
        print(f"Skipping {q} (already processed)")
        continue
        
    print(f"Processing {q}...")
    compressed_dir = COMPRESSED_BASE / f"cheng2020-attn_{q}"
    
    resnet_misclass = 0
    vit_misclass = 0
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
            if basename not in test_file_set:
                continue
                
            true_label = CLASSES.index(class_name)
            
            # Original image path - same naming pattern
            orig_img_path = ORIGINAL_DATA / class_name / filename.replace('.png', '.jpg')
            if not orig_img_path.exists():
                continue
            
            try:
                # Load images
                img = Image.open(img_path).convert('RGB')
                orig_img = Image.open(orig_img_path).convert('RGB')
                
                with torch.no_grad():
                    # ResNet evaluation
                    x_resnet = transform_resnet(img).unsqueeze(0).to(device)
                    x_orig_resnet = transform_resnet(orig_img).unsqueeze(0).to(device)
                    
                    pred_resnet = resnet(x_resnet).argmax(dim=1).item()
                    pred_orig_resnet = resnet(x_orig_resnet).argmax(dim=1).item()
                    
                    if pred_orig_resnet == true_label and pred_resnet != true_label:
                        resnet_misclass += 1
                    
                    # ViT evaluation
                    x_vit = transform_vit(img).unsqueeze(0).to(device)
                    x_orig_vit = transform_vit(orig_img).unsqueeze(0).to(device)
                    
                    pred_vit = vit(x_vit).argmax(dim=1).item()
                    pred_orig_vit = vit(x_orig_vit).argmax(dim=1).item()
                    
                    if pred_orig_vit == true_label and pred_vit != true_label:
                        vit_misclass += 1
                
                total_processed += 1
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
    
    total_misclass = resnet_misclass + vit_misclass
    results[q] = {
        'resnet': resnet_misclass,
        'vit': vit_misclass,
        'total': total_misclass,
        'processed': total_processed
    }
    print(f"  {q}: {total_misclass} misclassifications (ResNet: {resnet_misclass}, ViT: {vit_misclass}) from {total_processed} images\n")
    
    # Save after each quality level
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {OUTPUT_FILE}")

print("\n=== SUMMARY: EuroSAT Cheng2020 q1-q4 ===")
for q in ['q1', 'q2', 'q3', 'q4']:
    if q in results:
        print(f"{q}: {results[q]['total']}")
