#!/usr/bin/env python3
"""
Identify Misclassifications: Images correctly classified on originals but 
misclassified after neural compression.

Note: Not all misclassifications are "miscompressions" in the sense of 
Hofer & Böhme (2024). Manual inspection is required to distinguish between:
- Artifact-induced errors: Quality loss leads to misclassification
- True miscompressions: Semantic changes to image content

Focus: Neural codecs only (Cheng2020, MS-ILLM, JPEG-AI)

Output: CSV with all misclassified images for manual inspection

Features:
- Checkpoint support: Can resume after interruption
- Saves intermediate results after each codec/quality

Author: Vali Huter
Date: December 2025
"""

import sys
import json
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import timm

# Paths
PROJECT_ROOT = Path("/Users/vali/MA-Neural_Compression_Satellite_Images")
ORIGINAL_DATA = PROJECT_ROOT / "data" / "uncomp" / "EuroSAT_RGB"
COMPRESSED_NEURAL = PROJECT_ROOT / "data" / "compressed_neural"
SPLIT_FILE = PROJECT_ROOT / "data" / "uncomp" / "splits" / "rgb_test.json"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_classifier.pth"
VIT_MODEL_PATH = PROJECT_ROOT / "models" / "vit_classifier.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "miscompression_analysis"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.pkl"

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Only neural codecs - these can produce true "miscompressions"
NEURAL_CODECS = ['cheng2020-attn', 'msillm', 'jpeg_ai']

def load_resnet_model(model_path, device):
    """Load trained ResNet18 model"""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from utils.model_utils import BaselineClassifier
    
    model = BaselineClassifier(num_classes=10, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_vit_model(model_path, device):
    """Load trained ViT model"""
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_predictions_simple(model, data_root, image_list, transform, device, desc="Predicting"):
    """Get predictions for all images - simple version without batching for stability"""
    predictions = {}
    
    for img_path in tqdm(image_list, desc=desc):
        full_path = Path(data_root) / img_path
        
        # Try different extensions
        if not full_path.exists():
            full_path = full_path.with_suffix('.png')
        if not full_path.exists():
            full_path = full_path.with_suffix('.jpg')
        if not full_path.exists():
            continue
            
        try:
            image = Image.open(full_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, dim=1)
            
            predictions[img_path] = {
                'pred': int(pred.item()),
                'confidence': float(confidence.item()),
                'pred_class': CLASSES[int(pred.item())]
            }
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    return predictions

def get_ground_truth(image_list):
    """Extract ground truth labels from folder structure"""
    labels = {}
    for img_path in image_list:
        class_name = img_path.split('/')[0]
        labels[img_path] = {
            'label': CLASSES.index(class_name),
            'class_name': class_name
        }
    return labels

def find_neural_codec_qualities():
    """Find all quality levels for neural codecs"""
    codec_qualities = {}
    
    for codec in NEURAL_CODECS:
        codec_path = COMPRESSED_NEURAL / codec
        if not codec_path.exists():
            print(f"  Warning: {codec} not found at {codec_path}")
            continue
            
        qualities = []
        for item in codec_path.iterdir():
            if item.is_dir() and item.name.startswith(codec):
                # e.g., "cheng2020-attn_q1" -> "q1"
                q_name = item.name.replace(f"{codec}_", "")
                qualities.append((q_name, item))
        
        # Sort by quality level
        qualities.sort(key=lambda x: x[0])
        codec_qualities[codec] = qualities
    
    return codec_qualities

def save_checkpoint(checkpoint_data):
    """Save checkpoint for resuming"""
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  [Checkpoint saved]")

def load_checkpoint():
    """Load checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def save_intermediate_csv(miscompressions):
    """Save intermediate results to CSV"""
    csv_path = OUTPUT_DIR / "misclassifications.csv"
    with open(csv_path, 'w', newline='') as f:
        if miscompressions:
            writer = csv.DictWriter(f, fieldnames=miscompressions[0].keys())
            writer.writeheader()
            writer.writerows(miscompressions)
    print(f"  [Intermediate save: {len(miscompressions)} entries -> {csv_path.name}]")

def main():
    print("="*80)
    print("MISCLASSIFICATION IDENTIFICATION")
    print("Finding images misclassified after NEURAL compression")
    print("(Manual inspection required to identify true miscompressions)")
    print("="*80)
    
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n*** RESUMING from checkpoint ***")
        print(f"    Completed: {checkpoint['completed']}")
        all_miscompressions = checkpoint['miscompressions']
        completed_set = set(checkpoint['completed'])
        original_resnet = checkpoint['original_resnet']
        original_vit = checkpoint['original_vit']
        ground_truth = checkpoint['ground_truth']
    else:
        all_miscompressions = []
        completed_set = set()
        original_resnet = None
        original_vit = None
        ground_truth = None
    
    # Load test split
    with open(SPLIT_FILE, 'r') as f:
        image_list = json.load(f)
    print(f"Test images: {len(image_list)}")
    
    # Get ground truth if not from checkpoint
    if ground_truth is None:
        ground_truth = get_ground_truth(image_list)
    
    # Transforms
    resnet_transform = transforms.Compose([
        transforms.Resize((64, 64)),  # EuroSAT native size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT required size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load models
    print("\nLoading models...")
    resnet = load_resnet_model(MODEL_PATH, device)
    print("   ResNet-18 loaded")
    
    vit = load_vit_model(VIT_MODEL_PATH, device)
    print("   ViT-S/16 loaded")
    
    # Get baseline predictions on original images (if not from checkpoint)
    if original_resnet is None or original_vit is None:
        print("\n" + "="*60)
        print("STEP 1: Baseline predictions on original images")
        print("="*60)
        
        print("\nResNet-18 on originals...")
        original_resnet = get_predictions_simple(resnet, ORIGINAL_DATA, image_list, 
                                                  resnet_transform, device, "ResNet originals")
        
        print("\nViT on originals...")
        original_vit = get_predictions_simple(vit, ORIGINAL_DATA, image_list,
                                               vit_transform, device, "ViT originals")
        
        # Save checkpoint after baselines
        save_checkpoint({
            'completed': list(completed_set),
            'miscompressions': all_miscompressions,
            'original_resnet': original_resnet,
            'original_vit': original_vit,
            'ground_truth': ground_truth
        })
    else:
        print("\n  [Baseline predictions loaded from checkpoint]")
    
    # Count baseline accuracy
    resnet_correct = sum(1 for p in image_list if p in original_resnet and 
                         original_resnet[p]['pred'] == ground_truth[p]['label'])
    vit_correct = sum(1 for p in image_list if p in original_vit and 
                      original_vit[p]['pred'] == ground_truth[p]['label'])
    
    print(f"\nBaseline accuracy:")
    print(f"  ResNet-18: {resnet_correct}/{len(image_list)} ({100*resnet_correct/len(image_list):.2f}%)")
    print(f"  ViT-S/16:  {vit_correct}/{len(image_list)} ({100*vit_correct/len(image_list):.2f}%)")
    
    # Find neural codecs
    codec_qualities = find_neural_codec_qualities()
    print(f"\nFound neural codecs: {list(codec_qualities.keys())}")
    
    # Collect all miscompressions
    all_miscompressions = []
    
    # Process each neural codec
    for codec, qualities in codec_qualities.items():
        print("\n" + "="*60)
        print(f"STEP 2: Processing {codec}")
        print("="*60)
        
        for q_name, q_path in qualities:
            # Check if already completed
            task_id = f"{codec}_{q_name}"
            if task_id in completed_set:
                print(f"\n  Quality: {q_name} [SKIPPED - already completed]")
                continue
            
            print(f"\n  Quality: {q_name}")
            
            # ResNet predictions
            compressed_resnet = get_predictions_simple(
                resnet, q_path, image_list, resnet_transform, device,
                f"  ResNet {q_name}"
            )
            
            # ViT predictions  
            compressed_vit = get_predictions_simple(
                vit, q_path, image_list, vit_transform, device,
                f"  ViT {q_name}"
            )
            
            # Find miscompressions for ResNet
            for img_path in image_list:
                if img_path not in original_resnet or img_path not in compressed_resnet:
                    continue
                    
                orig_pred = original_resnet[img_path]['pred']
                comp_pred = compressed_resnet[img_path]['pred']
                true_label = ground_truth[img_path]['label']
                
                # Miscompression: correct on original, wrong after compression
                if orig_pred == true_label and comp_pred != true_label:
                    all_miscompressions.append({
                        'filename': img_path,
                        'model': 'ResNet-18',
                        'codec': codec,
                        'quality': q_name,
                        'true_class': ground_truth[img_path]['class_name'],
                        'original_pred': CLASSES[orig_pred],
                        'compressed_pred': CLASSES[comp_pred],
                        'original_confidence': original_resnet[img_path]['confidence'],
                        'compressed_confidence': compressed_resnet[img_path]['confidence'],
                        'original_correct': True,
                        'compressed_correct': False
                    })
            
            # Find miscompressions for ViT
            for img_path in image_list:
                if img_path not in original_vit or img_path not in compressed_vit:
                    continue
                    
                orig_pred = original_vit[img_path]['pred']
                comp_pred = compressed_vit[img_path]['pred']
                true_label = ground_truth[img_path]['label']
                
                if orig_pred == true_label and comp_pred != true_label:
                    all_miscompressions.append({
                        'filename': img_path,
                        'model': 'ViT-S/16',
                        'codec': codec,
                        'quality': q_name,
                        'true_class': ground_truth[img_path]['class_name'],
                        'original_pred': CLASSES[orig_pred],
                        'compressed_pred': CLASSES[comp_pred],
                        'original_confidence': original_vit[img_path]['confidence'],
                        'compressed_confidence': compressed_vit[img_path]['confidence'],
                        'original_correct': True,
                        'compressed_correct': False
                    })
            
            # Count for this quality
            resnet_miscomp = sum(1 for m in all_miscompressions 
                                 if m['model'] == 'ResNet-18' and m['codec'] == codec and m['quality'] == q_name)
            vit_miscomp = sum(1 for m in all_miscompressions 
                              if m['model'] == 'ViT-S/16' and m['codec'] == codec and m['quality'] == q_name)
            
            print(f"    Misclassifications: ResNet={resnet_miscomp}, ViT={vit_miscomp}")
            
            # Mark as completed and save checkpoint
            completed_set.add(task_id)
            save_checkpoint({
                'completed': list(completed_set),
                'miscompressions': all_miscompressions,
                'original_resnet': original_resnet,
                'original_vit': original_vit,
                'ground_truth': ground_truth
            })
            
            # Also save intermediate CSV
            save_intermediate_csv(all_miscompressions)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save CSV (final)
    csv_path = OUTPUT_DIR / "misclassifications.csv"
    with open(csv_path, 'w', newline='') as f:
        if all_miscompressions:
            writer = csv.DictWriter(f, fieldnames=all_miscompressions[0].keys())
            writer.writeheader()
            writer.writerows(all_miscompressions)
    
    print(f"\n Saved {len(all_miscompressions)} misclassifications to {csv_path}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\nTotal misclassifications found: {len(all_miscompressions)}")
    print("(Note: Manual inspection required to identify true miscompressions)")
    
    # By model
    by_model = defaultdict(int)
    for m in all_miscompressions:
        by_model[m['model']] += 1
    print("\nBy model:")
    for model, count in sorted(by_model.items()):
        print(f"  {model}: {count}")
    
    # By codec
    by_codec = defaultdict(int)
    for m in all_miscompressions:
        by_codec[m['codec']] += 1
    print("\nBy codec:")
    for codec, count in sorted(by_codec.items()):
        print(f"  {codec}: {count}")
    
    # By class confusion
    confusion = defaultdict(int)
    for m in all_miscompressions:
        key = f"{m['true_class']} -> {m['compressed_pred']}"
        confusion[key] += 1
    
    print("\nTop 10 class confusions:")
    for conf, count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"  {conf}: {count}")
    
    # Save summary JSON
    summary = {
        'total_miscompressions': len(all_miscompressions),
        'by_model': dict(by_model),
        'by_codec': dict(by_codec),
        'top_confusions': dict(sorted(confusion.items(), key=lambda x: -x[1])[:20])
    }
    
    summary_path = OUTPUT_DIR / "misclassification_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n Summary saved to {summary_path}")
    
    # Clean up checkpoint after successful completion
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(" Checkpoint removed (completed successfully)")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
