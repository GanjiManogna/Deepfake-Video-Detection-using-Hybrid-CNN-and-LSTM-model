#!/usr/bin/env python3
"""
GitHub predictor using their exact architecture and 97% accuracy model
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
# import face_recognition  # Not available, will use OpenCV fallback
from pathlib import Path

class GitHubModel(nn.Module):
    """Exact model architecture from GitHub repository"""
    
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(GitHubModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


class GitHubVideoPredictor:
    """Predictor using GitHub's exact implementation"""
    
    def __init__(self, model_path="saved_models/model_97_acc_100_frames_FF_data.pt", sequence_length: int | None = 24):
        import os
        try:
            from .device_utils import get_torch_device
            self.device, _ = get_torch_device()
        except Exception:
            self.device = torch.device('cpu')
        self.model_path = model_path
        # Speed-optimized default: use fewer frames unless explicitly overridden
        self.sequence_length = int(sequence_length) if sequence_length and sequence_length > 0 else 24
        # Temperature for calibration (reduce overconfidence of logits)
        try:
            self.temperature = float(os.getenv("LSTM_TEMPERATURE", "1.0"))
        except Exception:
            self.temperature = 1.0
        self.im_size = 112  # GitHub uses 112x112 images
        
        # Initialize model
        self.model = GitHubModel(2).to(self.device)
        self.load_model()
        self.model.eval()
        
        # Initialize transforms (exact from GitHub)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.im_size, self.im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Softmax for prediction
        self.sm = nn.Softmax(dim=1)
        
    def load_model(self):
        """Load the GitHub model"""
        if os.path.exists(self.model_path):
            print(f"Loading GitHub model from: {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✅ GitHub model loaded successfully")
        else:
            print(f"❌ Model not found at: {self.model_path}")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def _face_locations_safe(self, frame):
        """Face detection using OpenCV (GitHub's fallback method)"""
        try:
            # OpenCV face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            results = []
            for (x, y, w, h) in faces:
                top, right, bottom, left = y, x + w, y + h, x
                results.append((top, right, bottom, left))
            return results
        except Exception:
            return []
    
    def frame_extract(self, path):
        """Extract frames from video - exact from GitHub"""
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()
    
    def extract_frames_with_faces(self, video_path, max_frames=100):
        """Extract frames with face detection - GitHub's approach"""
        frames = []
        a = int(100 / self.sequence_length)  # GitHub's frame sampling
        first_frame = np.random.randint(0, a)
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            if i % a == first_frame:
                # Face detection and cropping
                faces = self._face_locations_safe(frame)
                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except:
                    pass  # Use full frame if no face detected
                
                frames.append(self.transform(frame))
                if len(frames) == self.sequence_length:
                    break
        
        # Pad if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        return frames
    
    def predict_video(self, video_path):
        """Predict using GitHub's exact method"""
        try:
            # Extract frames
            frames = self.extract_frames_with_faces(video_path, max_frames=100)
            
            if len(frames) == 0:
                return {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'error': 'No frames extracted from video'
                }
            
            # Create sequence tensor
            sequence = torch.stack(frames, dim=0).unsqueeze(0)  # [1, seq, 3, H, W]
            sequence = sequence.to(self.device)
            
            # Predict using GitHub's exact method
            with torch.no_grad():
                fmap, logits = self.model(sequence)
                # Apply temperature scaling before softmax if set (>0)
                if isinstance(self.temperature, float) and self.temperature > 0:
                    logits = logits / self.temperature
                logits = self.sm(logits)
                _, prediction = torch.max(logits, 1)
                confidence = logits[:, int(prediction.item())].item() * 100
            
            # GitHub's labeling: 0 -> REAL, 1 -> FAKE
            output = "FAKE" if prediction.item() == 1 else "REAL"
            
            return {
                'prediction': output,
                'confidence': confidence / 100.0,  # Convert to 0-1 range
                'raw_prediction': prediction.item(),
                'raw_confidence': confidence,
                'frames_processed': len(frames),
                'model': 'GitHub 97% Accuracy ResNeXt+LSTM',
                'sequence_length': self.sequence_length
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'error': str(e)
            }


def predict_video_github(video_path, sequence_length: int | None = 24):
    """Main prediction function using GitHub model (sequence_length default 24 for speed)."""
    predictor = GitHubVideoPredictor(sequence_length=sequence_length)
    return predictor.predict_video(video_path)


if __name__ == "__main__":
    # Test the GitHub predictor
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python github_predictor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = predict_video_github(video_path)
    
    print(f"Video: {video_path}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Raw Confidence: {result.get('raw_confidence', 'N/A')}%")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Frames processed: {result['frames_processed']}")
        print(f"Model: {result['model']}")
        print(f"Sequence length: {result.get('sequence_length', 'N/A')}")
