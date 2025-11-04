#!/usr/bin/env python3
"""
LSTM-only model for deepfake detection
Uses pre-extracted features and focuses on temporal patterns
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class LSTMOnlyModel(nn.Module):
    """LSTM-only model that processes temporal sequences of features"""
    
    def __init__(self, input_size=512, hidden_size=256, num_layers=2, 
                 bidirectional=True, dropout_rate=0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection (if needed)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # Output dimension
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Attention mechanism for temporal features
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x_flat = x.view(-1, self.input_size)  # (batch*seq_len, input_size)
        projected = self.input_projection(x_flat)  # (batch*seq_len, hidden_size)
        projected = projected.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(projected)  # (batch, seq_len, hidden_size*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(attended).squeeze(1)  # (batch,)
        
        return output

class LSTMOnlyPredictor:
    """Predictor class for LSTM-only model"""
    
    def __init__(self, model_path=None, device=None, threshold=0.5, 
                 input_size=512, seq_len=16):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.input_size = input_size
        self.seq_len = seq_len
        
        # Initialize model
        self.model = LSTMOnlyModel(input_size=input_size)
        self.model.to(self.device)
        
        # Load checkpoint if provided
        if model_path and Path(model_path).exists():
            self.load_checkpoint(model_path)
        
        # Feature extractor (simple CNN for basic features)
        self.feature_extractor = self._create_feature_extractor()
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
    
    def _create_feature_extractor(self):
        """Create a simple CNN feature extractor"""
        from torchvision import models
        
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            backbone = models.resnet18(pretrained=True)
        
        # Remove final layers
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc
        
        # Add custom layers to match input_size
        feature_extractor = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, self.input_size)  # ResNet18 output is 512
        )
        
        return feature_extractor
    
    def load_checkpoint(self, model_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded LSTM-only model from {model_path}")
        except Exception as e:
            print(f"Failed to load LSTM-only model: {e}")
    
    def extract_features(self, video_path, max_frames=16):
        """Extract features from video frames"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        features = []
        
        if total_frames == 0:
            cap.release()
            # Return zero features
            return torch.zeros(self.seq_len, self.input_size)
        
        # Sample frames
        indices = np.linspace(0, max(total_frames-1, 0), min(max_frames, self.seq_len), dtype=int)
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            if len(faces) > 0:
                # Use largest face
                x, y, w, h = max(faces, key=lambda x: x[2]*x[3])
                padding = int(min(w, h) * 0.25)
                x0 = max(0, x - padding)
                y0 = max(0, y - padding)
                x1 = min(rgb.shape[1], x + w + padding)
                y1 = min(rgb.shape[0], y + h + padding)
                crop = rgb[y0:y1, x0:x1]
            else:
                # Center crop if no face detected
                H, W = rgb.shape[:2]
                crop = rgb[H//4:3*H//4, W//4:3*W//4]
            
            if crop is None or crop.size == 0:
                crop = rgb
            
            # Resize and normalize
            crop = cv2.resize(crop, (224, 224))
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            
            # Convert to tensor
            tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)
            tensor = tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature = self.feature_extractor(tensor).squeeze(0)  # (input_size,)
                features.append(feature)
        
        cap.release()
        
        # Pad features if needed
        if len(features) == 0:
            features = [torch.zeros(self.input_size).to(self.device)] * self.seq_len
        while len(features) < self.seq_len:
            features.append(features[-1])
        
        # Stack features
        feature_sequence = torch.stack(features[:self.seq_len], dim=0)  # (seq_len, input_size)
        return feature_sequence
    
    def predict_video(self, video_path):
        """Predict if video is real or fake"""
        self.model.eval()
        
        try:
            features = self.extract_features(video_path)
            features = features.unsqueeze(0)  # (1, seq_len, input_size)
            features = features.to(self.device)
            
            with torch.no_grad():
                logits = self.model(features)
                probability = torch.sigmoid(logits).item()
                
                prediction = "FAKE" if probability > self.threshold else "REAL"
                confidence = probability if prediction == "FAKE" else 1.0 - probability
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probability": probability,
                    "frames_used": self.seq_len,
                    "model": "LSTM-Only (Temporal Features)",
                    "threshold": self.threshold
                }
        except Exception as e:
            return {
                "prediction": "ERROR",
                "error": str(e),
                "model": "LSTM-Only (Temporal Features)"
            }

# Global predictor instance
_lstm_predictor = None

def get_lstm_predictor(model_path=None, threshold=0.5):
    """Get LSTM-only predictor singleton"""
    global _lstm_predictor
    if _lstm_predictor is None:
        _lstm_predictor = LSTMOnlyPredictor(model_path, threshold=threshold)
    return _lstm_predictor

def predict_video_lstm_only(video_path, model_path=None, threshold=0.5):
    """Predict using LSTM-only model"""
    predictor = get_lstm_predictor(model_path, threshold)
    return predictor.predict_video(video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python lstm_only_model.py <video_path>")
        sys.exit(1)
    
    result = predict_video_lstm_only(sys.argv[1])
    print("LSTM-Only Result:", result)







