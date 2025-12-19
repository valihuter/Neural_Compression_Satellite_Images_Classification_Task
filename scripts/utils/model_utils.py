"""
Model utilities for loading trained classifiers

Contains the BaselineClassifier and EuroSATDataset classes
used by evaluation and analysis scripts.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
import json
from pathlib import Path

class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB"""
    
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]
    
    def __init__(
        self, 
        data_root: str, 
        split_file: str,
        transform=None
    ):
        """
        Args:
            data_root: Path to EuroSAT_RGB directory
            split_file: Path to split JSON (e.g., rgb_train.json)
            transform: Image transformations
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        
        with open(split_file, 'r') as f:
            self.image_paths = json.load(f)
        
        print(f"Loaded {len(self.image_paths)} images from {split_file}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_rel_path = self.image_paths[idx]
        img_path = self.data_root / img_rel_path
        
        image = Image.open(img_path).convert('RGB')
        
        class_name = img_rel_path.split('/')[0]
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class BaselineClassifier(nn.Module):
    """ResNet18-based classifier for EuroSAT"""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
