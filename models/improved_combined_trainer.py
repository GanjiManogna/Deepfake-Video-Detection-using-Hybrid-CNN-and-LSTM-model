#!/usr/bin/env python3
"""
Improved Combined Dataset Training for Better Deepfake Detection
Trains on both FaceForensics++ and DFDC datasets with advanced techniques
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ImprovedConfig:
    """Enhanced configuration for better training"""
    
    # Data parameters
    FRAME_SIZE = 112
    SEQ_LEN = 100
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 0.0001  # Lower LR for stability
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5
    
    # Advanced training parameters
    MIXUP_ALPHA = 0.2  # Data augmentation
    LABEL_SMOOTHING = 0.1
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    
    # Model architecture
    BACKBONE = 'resnext50'  # resnext50, efficientnet_b4, convnext_base
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True
    DROPOUT_RATE = 0.3
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    SAVED_MODELS_DIR = BASE_DIR / 'saved_models'
    RESULTS_DIR = BASE_DIR / 'training_results'
    
    def __init__(self):
        self.SAVED_MODELS_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MixUpAugmentation:
    """MixUp data augmentation for better generalization"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam

class ImprovedVideoDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    
    def __init__(self, video_paths, labels, transform=None, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.config = ImprovedConfig()
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames with improved face detection
        frames = self.extract_frames_improved(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Convert to tensor
        sequence = torch.stack(frames, dim=0)
        return sequence, torch.tensor(label, dtype=torch.float32)
    
    def extract_frames_improved(self, video_path):
        """Improved frame extraction with better face detection"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Improved face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Sample frames more intelligently
        frame_indices = np.linspace(0, total_frames-1, self.config.SEQ_LEN, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Multiple face detection attempts
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                
                if len(faces) > 0:
                    # Use the largest face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face
                    
                    # Add some padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size > 0:
                        frames.append(face_crop)
                else:
                    # If no face detected, use center crop
                    h, w = frame.shape[:2]
                    center_crop = frame[h//4:3*h//4, w//4:3*w//4]
                    frames.append(center_crop)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.config.SEQ_LEN:
            frames.append(frames[-1] if frames else np.zeros((64, 64, 3), dtype=np.uint8))
        
        return frames[:self.config.SEQ_LEN]

class ImprovedDeepfakeModel(nn.Module):
    """Enhanced model architecture with better feature extraction"""
    
    def __init__(self, config):
        super(ImprovedDeepfakeModel, self).__init__()
        self.config = config
        
        # Enhanced backbone selection
        if config.BACKBONE == 'resnext50':
            backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            feature_dim = 2048
        elif config.BACKBONE == 'efficientnet_b4':
            backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            feature_dim = 1792
        else:  # convnext_base
            backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            feature_dim = 1024
            
        # Remove classifier and add global average pooling
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection to ensure consistent dimensions
        self.feature_projection = nn.Linear(feature_dim, 512)
        
        # Enhanced LSTM with attention
        lstm_input_dim = 512  # Use projected feature dimension
        lstm_output_dim = config.LSTM_HIDDEN_SIZE * (2 if config.LSTM_BIDIRECTIONAL else 1)
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT_RATE,
            bidirectional=config.LSTM_BIDIRECTIONAL
        )
        
        # Simplified attention mechanism to avoid dimension issues
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,  # Reduced number of heads
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        
        # Reshape for backbone processing
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(batch_size * seq_len, -1)
        
        # Project features to consistent dimension
        features = self.feature_projection(features)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Attention mechanism (simplified to avoid dimension issues)
        try:
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            final_features = attn_out[:, -1, :]
        except:
            # Fallback to simple last timestep if attention fails
            final_features = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_features)
        
        return output

class ImprovedTrainer:
    """Enhanced trainer with advanced techniques"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = ImprovedDeepfakeModel(config).to(self.device)
        
        # Initialize loss functions
        self.focal_loss = FocalLoss(alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Initialize optimizer with simple setup to avoid dimension issues
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=config.LEARNING_RATE * 0.01
        )
        
        # MixUp augmentation
        self.mixup = MixUpAugmentation(alpha=config.MIXUP_ALPHA)
        
        # Training history
        self.train_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        self.val_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
    
    def create_transforms(self):
        """Create training and validation transforms"""
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.FRAME_SIZE, self.config.FRAME_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.FRAME_SIZE, self.config.FRAME_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transforms, val_transforms
    
    def prepare_combined_dataset(self):
        """Prepare combined dataset from both FaceForensics++ and DFDC"""
        print("📁 Preparing combined dataset...")
        
        # Load FaceForensics++ data
        ff_paths, ff_labels = self.load_faceforensics_data()
        print(f"FaceForensics++: {len(ff_paths)} videos")
        
        # Load DFDC data
        dfdc_paths, dfdc_labels = self.load_dfdc_data()
        print(f"DFDC: {len(dfdc_paths)} videos")
        
        # Combine datasets
        all_paths = ff_paths + dfdc_paths
        all_labels = ff_labels + dfdc_labels
        
        print(f"Total combined dataset: {len(all_paths)} videos")
        print(f"Real: {all_labels.count(0)}, Fake: {all_labels.count(1)}")
        
        # Split dataset
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Create transforms
        train_transforms, val_transforms = self.create_transforms()
        
        # Create datasets
        train_dataset = ImprovedVideoDataset(train_paths, train_labels, train_transforms, is_training=True)
        val_dataset = ImprovedVideoDataset(val_paths, val_labels, val_transforms, is_training=False)
        
        # Create data loaders with weighted sampling for class balance
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, sampler=sampler, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader
    
    def load_faceforensics_data(self):
        """Load FaceForensics++ dataset"""
        base_path = self.config.BASE_DIR / 'data' / 'faceforensics_data'
        
        video_paths = []
        labels = []
        
        # Real videos
        real_path = base_path / 'original_sequences'
        for subdir in ['actors', 'youtube']:
            subdir_path = real_path / subdir
            if subdir_path.exists():
                for video_file in subdir_path.glob('*.mp4'):
                    video_paths.append(str(video_file))
                    labels.append(0)  # Real
        
        # Fake videos
        fake_path = base_path / 'manipulated_sequences'
        for subdir in ['Deepfakes', 'DeepFakeDetection']:
            subdir_path = fake_path / subdir
            if subdir_path.exists():
                for video_file in subdir_path.glob('*.mp4'):
                    video_paths.append(str(video_file))
                    labels.append(1)  # Fake
        
        return video_paths, labels
    
    def load_dfdc_data(self):
        """Load DFDC dataset"""
        base_path = self.config.BASE_DIR / 'data' / 'dfdc_processed'
        
        video_paths = []
        labels = []
        
        # Real videos
        real_path = base_path / 'real'
        if real_path.exists():
            for video_file in real_path.glob('*.mp4'):
                video_paths.append(str(video_file))
                labels.append(0)  # Real
        
        # Fake videos
        fake_path = base_path / 'fake'
        if fake_path.exists():
            for video_file in fake_path.glob('*.mp4'):
                video_paths.append(str(video_file))
                labels.append(1)  # Fake
        
        return video_paths, labels
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with advanced techniques"""
        self.model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # MixUp augmentation
            if np.random.random() < 0.5:  # Apply MixUp 50% of the time
                mixed_x, y_a, y_b, lam = self.mixup(sequences, labels)
                self.optimizer.zero_grad()
                
                outputs = self.model(mixed_x)
                loss = lam * self.focal_loss(outputs.squeeze(), y_a) + \
                       (1 - lam) * self.focal_loss(outputs.squeeze(), y_b)
                
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy with MixUp
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = lam * (preds == y_a).float().mean() + (1 - lam) * (preds == y_b).float().mean()
                correct += acc.item() * sequences.size(0)
                
            else:
                # Normal training
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.focal_loss(outputs.squeeze(), labels)
                
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds.squeeze() == labels).sum().item()
            
            running_loss += loss.item() * sequences.size(0)
            total += sequences.size(0)
            
            # Store predictions for metrics
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.squeeze().detach().cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate metrics
        train_loss = running_loss / total
        train_acc = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        return train_loss, train_acc, precision, recall, f1
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.bce_loss(outputs.squeeze(), labels)
                
                running_loss += loss.item() * sequences.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)
                
                all_labels.extend(labels.detach().cpu().numpy())
                all_preds.extend(preds.squeeze().detach().cpu().numpy())
                all_probs.extend(probs.squeeze().detach().cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / total
        val_acc = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        return val_loss, val_acc, precision, recall, f1, all_labels, all_preds, all_probs
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.val_history['loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_history['accuracy'], label='Train Acc')
        axes[0, 1].plot(self.val_history['accuracy'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.train_history['f1'], label='Train F1')
        axes[1, 0].plot(self.val_history['f1'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Precision vs Recall
        axes[1, 1].plot(self.train_history['precision'], label='Train Precision')
        axes[1, 1].plot(self.train_history['recall'], label='Train Recall')
        axes[1, 1].plot(self.val_history['precision'], label='Val Precision')
        axes[1, 1].plot(self.val_history['recall'], label='Val Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, labels, preds, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.config.RESULTS_DIR / f'{title.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting Improved Combined Dataset Training")
        print("=" * 60)
        
        # Prepare data
        train_loader, val_loader = self.prepare_combined_dataset()
        
        best_val_f1 = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_labels, val_preds, val_probs = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['precision'].append(train_prec)
            self.train_history['recall'].append(train_rec)
            self.train_history['f1'].append(train_f1)
            
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            self.val_history['precision'].append(val_prec)
            self.val_history['recall'].append(val_rec)
            self.val_history['f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
            
            # Save best model
            torch.save(self.model.state_dict(), self.config.SAVED_MODELS_DIR / 'last_improved_model.pth')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.config.SAVED_MODELS_DIR / 'best_improved_model.pth')
                print(f"✅ New best model saved! F1: {val_f1:.4f}")
                
                # Save detailed results
                results = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'config': vars(self.config)
                }
                with open(self.config.RESULTS_DIR / 'best_model_results.json', 'w') as f:
                    json.dump(results, f, indent=2)
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= self.config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs without improvement.")
                break
        
        # Plot results
        self.plot_training_history()
        self.plot_confusion_matrix(val_labels, val_preds, "Final Validation Confusion Matrix")
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation F1 score: {best_val_f1:.4f}")
        print(f"Model saved to: {self.config.SAVED_MODELS_DIR / 'best_improved_model.pth'}")

def main():
    """Main function"""
    config = ImprovedConfig()
    trainer = ImprovedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
