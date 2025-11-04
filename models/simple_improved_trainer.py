#!/usr/bin/env python3
"""
simple_improved_trainer.py
Stable trainer for ResNeXt50 + LSTM deepfake detector.

Usage:
    python simple_improved_trainer.py
Saves:
    saved_models/best_simple_model.pth
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
from PIL import Image
import sklearn.metrics as skm
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
class SimpleConfig:
    FRAME_SIZE = 160
    SEQ_LEN = 16
    BATCH_SIZE = 2
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5

    # Switch default backbone to EfficientNet-B0 to match trained hybrid
    BACKBONE = 'efficientnet_b0'  # was 'resnext50'
    # Match checkpoint dims: projection -> 1024, LSTM hidden 512, BiLSTM => 1024 output
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 1
    LSTM_BIDIRECTIONAL = True
    DROPOUT_RATE = 0.3

    SPATIAL_ATTENTION = False
    TEMPORAL_ATTENTION = False
    MULTIMODAL_FUSION = False

    USE_MIXED_PRECISION = False
    GRADIENT_CLIP = 1.0
    WARMUP_EPOCHS = 2

    BASE_DIR = Path(__file__).resolve().parent
    SAVED_MODELS_DIR = BASE_DIR / "saved_models"
    RESULTS_DIR = BASE_DIR / "training_results"

    def __init__(self):
        self.SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Model
# ----------------------------
class SimpleDeepfakeModel(nn.Module):
    """ResNeXt50 backbone -> projection -> LSTM -> classifier"""
    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config

        # optional attention imports (safe)
        try:
            from models.attention import TemporalAttention, SpatialAttention, MultiModalAttention
        except Exception:
            TemporalAttention = SpatialAttention = MultiModalAttention = None

        # Load backbone based on config
        backbone_out_dim = 2048
        if (self.config.BACKBONE or '').lower() == 'efficientnet_b0':
            try:
                eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            except Exception:
                eff = models.efficientnet_b0(pretrained=True)
            # Use feature extractor block
            self.backbone = eff.features
            backbone_out_dim = 1280
        else:
            try:
                backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnext50_32x4d(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_out_dim = 2048

        # Attention placeholders (not used by default)
        self.spatial_attention = SpatialAttention(2048) if (config.SPATIAL_ATTENTION and SpatialAttention) else None

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # LSTM projected dim
        self.lstm_output_dim = config.LSTM_HIDDEN_SIZE * (2 if config.LSTM_BIDIRECTIONAL else 1)

        # Project 2048 -> lstm_output_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, self.lstm_output_dim),
            nn.BatchNorm1d(self.lstm_output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_output_dim,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=config.LSTM_BIDIRECTIONAL,
            dropout=config.DROPOUT_RATE if config.LSTM_LAYERS > 1 else 0.0
        )

        self.temporal_attention = TemporalAttention(self.lstm_output_dim) if (config.TEMPORAL_ATTENTION and TemporalAttention) else None

        classifier_input_dim = self.lstm_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.backbone(x)  # expect (B*seq_len, 2048, 1, 1) or similar

        if self.spatial_attention is not None:
            spatial_features, _ = self.spatial_attention(spatial_features)

        pooled = self.global_pool(spatial_features).view(batch_size * seq_len, -1)
        projected = self.feature_projection(pooled)  # (B*seq_len, lstm_output_dim)
        features = projected.view(batch_size, seq_len, -1)  # (batch, seq_len, feat)

        lstm_out, _ = self.lstm(features)  # (batch, seq_len, hidden*directions)

        if self.temporal_attention is not None:
            lstm_out, _ = self.temporal_attention(lstm_out)
        else:
            if self.config.LSTM_BIDIRECTIONAL:
                forward_last = lstm_out[:, -1, :self.config.LSTM_HIDDEN_SIZE]
                backward_last = lstm_out[:, 0, self.config.LSTM_HIDDEN_SIZE:]
                lstm_out = torch.cat([forward_last, backward_last], dim=1)
            else:
                lstm_out = lstm_out[:, -1, :]  # last time-step

        logits = self.classifier(lstm_out).squeeze(1)  # (batch,)
        return logits

# ----------------------------
# Dataset
# ----------------------------
class SimpleVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, seq_len=16, frame_size=160):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.seq_len = seq_len
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.extract_frames_simple(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        sequence = torch.stack(frames, dim=0)  # (seq_len, C, H, W)
        return sequence, torch.tensor(label, dtype=torch.float32)

    def extract_frames_simple(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        if total_frames == 0:
            cap.release()
            # return black frames
            return [Image.fromarray(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))] * self.seq_len

        # sample indices evenly
        frame_indices = np.linspace(0, max(total_frames - 1, 0), self.seq_len, dtype=int)

        # haarface fallback
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret or frame is None:
                # pad with last good frame or black
                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                face_crop = frame_rgb[y:y+h, x:x+w]
                pil = Image.fromarray(face_crop)
            else:
                # center crop
                H, W = frame_rgb.shape[:2]
                h0, w0 = int(H * 0.5), int(W * 0.5)
                y0, x0 = H//2 - h0//2, W//2 - w0//2
                center_crop = frame_rgb[y0:y0+h0, x0:x0+w0]
                pil = Image.fromarray(center_crop)

            pil = pil.resize((self.frame_size, self.frame_size))
            frames.append(pil)

        cap.release()

        # pad if necessary
        if not frames:
            frames = [Image.fromarray(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))] * self.seq_len
        while len(frames) < self.seq_len:
            frames.append(frames[-1])
        return frames[:self.seq_len]

# ----------------------------
# Trainer
# ----------------------------
class SimpleTrainer:
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model = SimpleDeepfakeModel(config).to(self.device)

        # placeholder criterion (will set pos_weight after dataset is prepared)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

    def create_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        return train_transforms, val_transforms

    def load_faceforensics_data(self, max_videos=200):
        base = self.config.BASE_DIR / "data" / "faceforensics_data"
        video_paths, labels = [], []
        real_path = base / "original_sequences"
        if real_path.exists():
            for sub in ["actors", "youtube"]:
                p = real_path / sub
                if p.exists():
                    vids = list(p.glob("*.mp4"))[: max_videos//4]
                    for v in vids:
                        video_paths.append(str(v))
                        labels.append(0)
        fake_path = base / "manipulated_sequences"
        if fake_path.exists():
            for sub in ["Deepfakes", "DeepFakeDetection"]:
                p = fake_path / sub
                if p.exists():
                    vids = list(p.glob("*.mp4"))[: max_videos//4]
                    for v in vids:
                        video_paths.append(str(v))
                        labels.append(1)
        return video_paths, labels

    def load_dfdc_data(self, max_videos=200):
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
        print("Preparing dataset...")
        ff_paths, ff_labels = self.load_faceforensics_data(max_videos_per_dataset)
        dfdc_paths, dfdc_labels = self.load_dfdc_data(max_videos_per_dataset)

        all_paths = ff_paths + dfdc_paths
        all_labels = ff_labels + dfdc_labels

        print(f"Total videos: {len(all_paths)} (Real: {all_labels.count(0)}, Fake: {all_labels.count(1)})")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

        train_t, val_t = self.create_transforms()
        train_dataset = SimpleVideoDataset(train_paths, train_labels, transform=train_t,
                                           seq_len=self.config.SEQ_LEN, frame_size=self.config.FRAME_SIZE)
        val_dataset = SimpleVideoDataset(val_paths, val_labels, transform=val_t,
                                         seq_len=self.config.SEQ_LEN, frame_size=self.config.FRAME_SIZE)

        # Weighted sampling to balance classes in training
        counts = Counter(train_labels)
        num_real = counts.get(0, 0)
        num_fake = counts.get(1, 0)
        print("Train split counts:", counts)
        # avoid divide-by-zero
        weight_real = 1.0 / (num_real + 1e-6)
        weight_fake = 1.0 / (num_fake + 1e-6)
        sample_weights = [weight_real if l == 0 else weight_fake for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)

        # Set loss pos_weight to handle imbalance (torch expects pos_weight for positive class)
        if num_fake == 0:
            pos_weight = torch.tensor([1.0], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([num_real / (num_fake + 1e-6)], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        print("pos_weight set to:", pos_weight.item())

        return train_loader, val_loader

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for sequences, labels in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            # optional gradient clip
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def validate_epoch(self, val_loader, threshold=0.5):
        self.model.eval()
        all_preds, all_targets = [], []
        running_loss, total = 0.0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
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
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", report)
        return val_loss, val_acc

    def train(self):
        train_loader, val_loader = self.prepare_dataset()
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader, threshold=0.5)
            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # save last
            torch.save(self.model.state_dict(), self.config.SAVED_MODELS_DIR / "last_simple_model.pth")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.config.SAVED_MODELS_DIR / "best_simple_model.pth")
                print("✅ New best model saved!")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.PATIENCE:
                print("Early stopping.")
                break

        print("Training finished. Best val loss:", best_val_loss)

# ----------------------------
# Main
# ----------------------------
def main():
    config = SimpleConfig()
    trainer = SimpleTrainer(config)

    # --- Evaluate on test set after training ---
    # Prepare a real test split (20% of all data, not used in train/val)
    ff_paths, ff_labels = trainer.load_faceforensics_data()
    dfdc_paths, dfdc_labels = trainer.load_dfdc_data()
    all_paths = ff_paths + dfdc_paths
    all_labels = ff_labels + dfdc_labels
    # Split: 64% train, 16% val, 20% test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    # Use same transform as validation
    _, val_t = trainer.create_transforms()
    test_dataset = SimpleVideoDataset(test_paths, test_labels, transform=val_t, seq_len=config.SEQ_LEN, frame_size=config.FRAME_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    trainer.model.eval()
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(trainer.device)
            labels = labels.to(trainer.device)
            outputs = trainer.model(sequences)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().long().tolist())

    print("\nTest Classification Report:\n", skm.classification_report(all_targets, all_preds, digits=4))
    print(f"Test Accuracy: {skm.accuracy_score(all_targets, all_preds):.4f}")

if __name__ == "__main__":
    main()





