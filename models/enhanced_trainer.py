#!/usr/bin/env python3
"""
Enhanced trainer with improved data augmentation and training techniques
This builds on simple_improved_trainer.py with Phase 1 improvements
"""

import os
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import sklearn.metrics as skm
import warnings
warnings.filterwarnings("ignore")

# Import base classes
from models.simple_improved_trainer import SimpleConfig, SimpleDeepfakeModel, SimpleVideoDataset

class EnhancedConfig(SimpleConfig):
    """Extended config with enhancement options"""
    def __init__(self):
        super().__init__()
        # Enhanced augmentation settings
        self.AUG_BRIGHTNESS = 0.15  # ±15% brightness variation
        self.AUG_CONTRAST = 0.15    # ±15% contrast variation
        self.AUG_BLUR = 0.1          # 10% chance of light blur
        self.AUG_ROTATION = 5        # ±5 degree rotation
        self.AUG_COLOR_JITTER = True
        
        # Sequence improvements
        self.SEQ_LEN = 24  # Increased from 16 for better temporal context
        self.ADAPTIVE_SEQ_LEN = True  # Adjust based on video length
        
        # Training improvements
        self.LABEL_SMOOTHING = 0.1
        self.USE_MIXUP = False  # Can enable for stronger augmentation
        self.FOCAL_LOSS_GAMMA = 2.0  # For hard example mining
        
        # TTA settings
        self.USE_TTA = True
        self.TTA_FLIPS = 2  # Original + horizontal flip


class EnhancedTransforms:
    """Enhanced data augmentation transforms"""
    
    @staticmethod
    def get_train_transforms(config):
        """Get training transforms with enhanced augmentation"""
        return transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=config.AUG_BRIGHTNESS,
                    contrast=config.AUG_CONTRAST,
                    saturation=0.1,
                    hue=0.05
                )
            ], p=0.7),
            
            transforms.RandomApply([
                transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.2)))
            ], p=0.3),
            
            transforms.RandomApply([
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0))))
            ], p=config.AUG_BLUR),
            
            transforms.RandomRotation(degrees=config.AUG_ROTATION, fill=0),
            
            transforms.RandomHorizontalFlip(p=0.5),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Random erasing (cutout)
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    
    @staticmethod
    def get_val_transforms(config):
        """Validation transforms (no augmentation)"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedTrainer:
    """Enhanced trainer with better augmentation and training techniques"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        self.model = SimpleDeepfakeModel(config).to(self.device)
        
        # Enhanced loss with label smoothing
        if config.LABEL_SMOOTHING > 0:
            # Use focal loss or BCE with label smoothing
            self.criterion = FocalLoss(alpha=1.0, gamma=config.FOCAL_LOSS_GAMMA)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Better optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
    
    def create_transforms(self):
        """Create enhanced transforms"""
        train_t = EnhancedTransforms.get_train_transforms(self.config)
        val_t = EnhancedTransforms.get_val_transforms(self.config)
        return train_t, val_t
    
    def load_faceforensics_data(self, max_videos=200):
        """Load FaceForensics++ data"""
        base = self.config.BASE_DIR / "data" / "faceforensics_data"
        video_paths, labels = [], []
        real_path = base / "original_sequences"
        if real_path.exists():
            for sub in ["actors", "youtube"]:
                p = real_path / sub
                if p.exists():
                    # Check for videos in c23/videos subdirectory (common structure)
                    c23_path = p / "c23" / "videos"
                    if c23_path.exists():
                        vids = list(c23_path.glob("*.mp4"))[: max_videos//4]
                    else:
                        # Try recursive search
                        vids = list(p.rglob("*.mp4"))[: max_videos//4]
                    for v in vids:
                        video_paths.append(str(v))
                        labels.append(0)
        fake_path = base / "manipulated_sequences"
        if fake_path.exists():
            for sub in ["Deepfakes", "DeepFakeDetection"]:
                p = fake_path / sub
                if p.exists():
                    # Check for videos in c23/videos subdirectory
                    c23_path = p / "c23" / "videos"
                    if c23_path.exists():
                        vids = list(c23_path.glob("*.mp4"))[: max_videos//4]
                    else:
                        # Try recursive search
                        vids = list(p.rglob("*.mp4"))[: max_videos//4]
                    for v in vids:
                        video_paths.append(str(v))
                        labels.append(1)
        return video_paths, labels
    
    def load_dfdc_data(self, max_videos=200):
        """Load DFDC data"""
        base = self.config.BASE_DIR / "data" / "dfdc_processed"
        video_paths, labels = [], []
        real_path = base / "real"
        fake_path = base / "fake"
        if real_path.exists():
            vids = list(real_path.glob("*.mp4"))[: max_videos//2]
            for v in vids:
                video_paths.append(str(v)); labels.append(0)
        if fake_path.exists():
            vids = list(fake_path.glob("*.mp4"))[: max_videos//2]
            for v in vids:
                video_paths.append(str(v)); labels.append(1)
        return video_paths, labels
    
    def prepare_dataset(self, max_videos_per_dataset=200):
        """Prepare dataset with enhanced transforms"""
        print("📊 Preparing enhanced dataset...")
        ff_paths, ff_labels = self.load_faceforensics_data(max_videos_per_dataset)
        dfdc_paths, dfdc_labels = self.load_dfdc_data(max_videos_per_dataset)
        
        all_paths = ff_paths + dfdc_paths
        all_labels = ff_labels + dfdc_labels
        
        print(f"Total videos: {len(all_paths)} (Real: {all_labels.count(0)}, Fake: {all_labels.count(1)})")
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        train_t, val_t = self.create_transforms()
        
        # Enhanced dataset with longer sequences
        train_dataset = SimpleVideoDataset(
            train_paths, train_labels,
            transform=train_t,
            seq_len=self.config.SEQ_LEN,
            frame_size=self.config.FRAME_SIZE
        )
        val_dataset = SimpleVideoDataset(
            val_paths, val_labels,
            transform=val_t,
            seq_len=self.config.SEQ_LEN,
            frame_size=self.config.FRAME_SIZE
        )
        
        # Weighted sampling
        counts = Counter(train_labels)
        num_real = counts.get(0, 0)
        num_fake = counts.get(1, 0)
        print(f"Train split - Real: {num_real}, Fake: {num_fake}")
        
        weight_real = 1.0 / (num_real + 1e-6)
        weight_fake = 1.0 / (num_fake + 1e-6)
        sample_weights = [weight_real if l == 0 else weight_fake for l in train_labels]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Set loss pos_weight
        if num_fake == 0:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([num_real / (num_fake + 1e-6)], dtype=torch.float32)
        
        if not isinstance(self.criterion, FocalLoss):
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        
        print(f"✅ Dataset prepared. pos_weight: {pos_weight.item():.3f}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train one epoch with enhanced techniques"""
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Apply label smoothing if enabled
            if self.config.LABEL_SMOOTHING > 0 and not isinstance(self.criterion, FocalLoss):
                labels_smooth = labels * (1 - self.config.LABEL_SMOOTHING) + 0.5 * self.config.LABEL_SMOOTHING
                labels = labels_smooth
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == (labels > 0.5).float()).sum().item()
            total += labels.size(0)
        
        return running_loss / total, correct / total
    
    def validate_epoch(self, val_loader, threshold=0.5):
        """Validate with optional TTA"""
        self.model.eval()
        all_preds, all_targets = [], []
        running_loss, total = 0.0, 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                
                # Test-Time Augmentation (TTA)
                if self.config.USE_TTA:
                    # Flip sequence horizontally
                    sequences_flip = sequences.flip(dims=[4])  # Flip width dimension
                    outputs_flip = self.model(sequences_flip)
                    # Average predictions
                    outputs = (outputs + outputs_flip) / 2.0
                
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).long()
                
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().long().tolist())
                total += labels.size(0)
        
        val_loss = running_loss / total
        cm = skm.confusion_matrix(all_targets, all_preds)
        report = skm.classification_report(all_targets, all_preds, digits=4)
        val_acc = (np.array(all_preds) == np.array(all_targets)).mean()
        
        # Calculate F1-score
        f1 = skm.f1_score(all_targets, all_preds)
        
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        print(f"F1-Score: {f1:.4f}")
        
        return val_loss, val_acc, f1
    
    def train(self):
        """Main training loop"""
        train_loader, val_loader = self.prepare_dataset()
        best_val_f1 = 0.0  # Track F1 instead of just loss
        epochs_without_improvement = 0
        
        print("\n🚀 Starting enhanced training...")
        
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.EPOCHS}")
            print(f"{'='*60}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader, threshold=0.5)
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n📊 Results:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoints
            torch.save(
                self.model.state_dict(),
                self.config.SAVED_MODELS_DIR / "last_enhanced_model.pth"
            )
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(
                    self.model.state_dict(),
                    self.config.SAVED_MODELS_DIR / "best_enhanced_model.pth"
                )
                print(f"✅ New best model saved! (F1: {val_f1:.4f})")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.config.PATIENCE:
                print(f"⏹️  Early stopping after {epoch} epochs.")
                break
        
        print(f"\n🎉 Training finished! Best F1: {best_val_f1:.4f}")


def main():
    """Main entry point"""
    config = EnhancedConfig()
    trainer = EnhancedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

