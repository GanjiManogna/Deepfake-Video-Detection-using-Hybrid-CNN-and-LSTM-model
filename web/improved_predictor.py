#!/usr/bin/env python3
"""
improved_predictor.py

Safe inference loader for your deepfake model checkpoint.
- Tries to import your SimpleDeepfakeModel / SimpleConfig from models/simple_improved_trainer.py
- Loads checkpoint using strict=False (so mismatched sizes/extra layers won't crash)
- Prints missing/unexpected keys so you can inspect what is/was different
- Uses a safe default threshold (0.6) that you can adjust
- Exposes predict_video_improved(video_path) for your app.py / flask endpoint

Usage (quick):
    from web.improved_predictor import predict_video_improved
    result = predict_video_improved("/path/to/video.mp4")
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# allow importing from models folder
sys.path.append(str(Path(__file__).resolve().parent.parent / "models"))

# Default config fallback
DEFAULT_SEQ_LEN = 16  # Reduced for faster inference
DEFAULT_FRAME_SIZE = 160
# Slightly conservative threshold to avoid false FAKE on real videos
DEFAULT_THRESHOLD = 0.60
# Support both original and enhanced models
_USE_ENHANCED = os.getenv("USE_ENHANCED_MODEL", "0") == "1"
if _USE_ENHANCED:
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "best_enhanced_model.pth"
    # Fallback to original if enhanced not found
    if not DEFAULT_MODEL_PATH.exists():
        DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "best_simple_model.pth"
else:
    DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "best_simple_model.pth"

# Try to import your SimpleConfig and SimpleDeepfakeModel from the trainer if present
SimpleConfig = None
SimpleDeepfakeModel = None
try:
    from simple_improved_trainer import SimpleConfig, SimpleDeepfakeModel  # if models/ placed in sys.path
except Exception:
    try:
        # maybe the file is named different or is in top-level models package
        from models.simple_improved_trainer import SimpleConfig, SimpleDeepfakeModel
    except Exception:
        SimpleConfig = None
        SimpleDeepfakeModel = None

# If not available, define a minimal compatible model structure (ResNeXt50 -> proj -> LSTM -> classifier)
class FallbackModel(nn.Module):
    def __init__(self, seq_len=DEFAULT_SEQ_LEN, frame_size=DEFAULT_FRAME_SIZE, lstm_hidden=256, lstm_layers=1, bidir=False):
        super().__init__()
        # load resnext50 backbone (no weights required here)
        try:
            backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnext50_32x4d(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove final fc
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, lstm_hidden * (2 if bidir else 1))
        self.lstm = nn.LSTM(input_size=lstm_hidden * (2 if bidir else 1),
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=bidir)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * (2 if bidir else 1), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, seq, C, H, W)
        b, seq, c, h, w = x.shape
        x = x.view(b * seq, c, h, w)
        features = self.backbone(x)
        features = self.global_pool(features).view(b * seq, -1)
        proj = self.proj(features)  # (b*seq, feat)
        proj = proj.view(b, seq, -1)
        lstm_out, _ = self.lstm(proj)
        out = lstm_out[:, -1, :]
        logits = self.classifier(out).squeeze(1)
        return logits

# Predictor class
class SafeVideoPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD, device=None):
        self.model_path = Path(model_path)
        self.threshold = float(threshold)
        if device is not None:
            self.device = device
        else:
            try:
                from .device_utils import get_torch_device
                self.device, _ = get_torch_device()
            except Exception:
                self.device = torch.device("cpu")
        # Temperature < 1.0 sharpens probabilities; helps low-confidence hybrids
        self.temperature = 0.7

        # config values (prefer your SimpleConfig if available)
        if SimpleConfig is not None:
            try:
                cfg = SimpleConfig()
                # Prefer env override, else at least 24 frames for stronger hybrid signal
                env_seq = os.getenv("HYBRID_SEQ_LEN")
                if env_seq and str(env_seq).isdigit():
                    self.seq_len = int(env_seq)
                else:
                    self.seq_len = max(16, getattr(cfg, "SEQ_LEN", DEFAULT_SEQ_LEN))  # Reduced minimum
                self.frame_size = getattr(cfg, "FRAME_SIZE", DEFAULT_FRAME_SIZE)
            except Exception:
                self.seq_len = max(16, DEFAULT_SEQ_LEN)  # Reduced minimum
                self.frame_size = DEFAULT_FRAME_SIZE
        else:
            self.seq_len = max(16, DEFAULT_SEQ_LEN)  # Reduced minimum
            self.frame_size = DEFAULT_FRAME_SIZE

        # build model instance
        if SimpleDeepfakeModel is not None:
            try:
                cfg = SimpleConfig() if SimpleConfig is not None else None
                if cfg is not None:
                    self.model = SimpleDeepfakeModel(cfg)
                else:
                    self.model = SimpleDeepfakeModel(SimpleConfig())  # try anyway
            except Exception:
                # fallback to FallbackModel
                self.model = FallbackModel(seq_len=self.seq_len, frame_size=self.frame_size)
        else:
            self.model = FallbackModel(seq_len=self.seq_len, frame_size=self.frame_size)

        self.model.to(self.device)
        # Try loading learned temperature if available
        try:
            temp_path = Path(__file__).resolve().parent.parent / "training_results" / "temperature.json"
            if temp_path.exists():
                import json
                with open(temp_path, "r", encoding="utf-8") as f:
                    tval = json.load(f).get("temperature")
                    if isinstance(tval, (int, float)) and 0.2 <= float(tval) <= 5.0:
                        self.temperature = float(tval)
        except Exception:
            pass
        self._load_checkpoint_strict_false()

        # transforms (must match training normalisation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.frame_size, self.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def _load_checkpoint_strict_false(self):
        if not self.model_path.exists():
            print(f"⚠️ Model checkpoint not found at {self.model_path}. The model will run with random init (poor results).")
            return

        checkpoint = torch.load(self.model_path, map_location=self.device)
        # checkpoint may be a dict (with keys like 'model_state_dict', 'state_dict') or a raw state_dict
        state_dict = None
        if isinstance(checkpoint, dict):
            # Common keys to look for
            possible_keys = ['model_state_dict', 'state_dict', 'state-dict', 'model']
            for k in possible_keys:
                if k in checkpoint:
                    state_dict = checkpoint[k]
                    break
            if state_dict is None:
                # maybe checkpoint is the state_dict itself, or other structure. Try to use the dict as state_dict if keys look like tensors
                # We'll assume checkpoint is a state_dict if its values are tensors
                if all(hasattr(v, "shape") for v in checkpoint.values()):
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict is None:
            print("⚠️ Couldn't find a workable state_dict in checkpoint. Attempting to load entire checkpoint with strict=False.")
            try:
                self.model.load_state_dict(checkpoint, strict=False)
                print("Loaded checkpoint (best-effort).")
            except Exception as e:
                print("Failed to load checkpoint:", e)
            return

        # Try a strict=False load and print missing/unexpected keys
        try:
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            # PyTorch returns an OrderedDict when strict=False in some versions; handle both
            if isinstance(missing, (list, tuple)) or isinstance(unexpected, (list, tuple)):
                print(f"Loaded checkpoint with strict=False. Missing keys: {missing}. Unexpected keys: {unexpected}.")
            else:
                # Older PyTorch returns None but raises errors otherwise. We'll still print a small message.
                print("Loaded checkpoint with strict=False (PyTorch did not return missing/unexpected lists).")
        except TypeError:
            # Older PyTorch versions return a NamedTuple; call load_state_dict and get return values differently
            try:
                result = self.model.load_state_dict(state_dict, strict=False)
                print("Loaded checkpoint (returned):", result)
            except Exception as e:
                print("Error while loading checkpoint with strict=False:", e)
        except Exception as e:
            print("Error while loading checkpoint:", e)

    def extract_frames(self, video_path):
        """Extract seq_len frames with face detection fallback center crops. Returns list of PIL images / tensors."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames = []

        if total_frames == 0:
            cap.release()
            # return blank frames
            blank = Image.fromarray(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
            return [self.transform(np.array(blank)) for _ in range(self.seq_len)]

        indices = np.linspace(0, max(total_frames-1, 0), self.seq_len, dtype=int)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40,40))  # Faster detection
            if len(faces) > 0:
                x,y,w,h = max(faces, key=lambda x: x[2]*x[3])
                padding = int(min(w,h) * 0.15)  # Reduced padding for speed
                x0 = max(0, x-padding); y0 = max(0, y-padding)
                x1 = min(rgb.shape[1], x + w + padding); y1 = min(rgb.shape[0], y + h + padding)
                crop = rgb[y0:y1, x0:x1]
            else:
                H,W = rgb.shape[:2]
                crop = rgb[H//4:3*H//4, W//4:3*W//4]
            # ensure not empty
            if crop is None or crop.size == 0:
                crop = rgb
            try:
                tensor = self.transform(crop)
            except Exception:
                # fallback: convert to PIL then transform
                tensor = self.transform(Image.fromarray(crop))
            frames.append(tensor)

        cap.release()

        # pad frames
        if len(frames) == 0:
            blank = torch.zeros(3, self.frame_size, self.frame_size)
            frames = [blank] * self.seq_len
        while len(frames) < self.seq_len:
            frames.append(frames[-1])
        return frames[:self.seq_len]

    def predict_video(self, video_path, threshold=None):
        """Run inference on a local video file. Returns a dict with prediction and prob"""
        threshold = self.threshold if threshold is None else float(threshold)
        self.model.eval()

        try:
            frames = self.extract_frames(video_path)
            seq = torch.stack(frames, dim=0).unsqueeze(0)  # (1, seq_len, C, H, W)
            seq = seq.to(self.device)

            with torch.no_grad():
                output = self.model(seq)
                # model might return logits or (logits, attn). Handle both.
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output
                # logits shape (1,) or (1,1) etc.
                # Temperature scaling to sharpen probabilities
                probs = torch.sigmoid(logits / self.temperature)
                prob = float(probs.view(-1)[0].cpu().item())
                # Fix: Higher probability should mean FAKE, lower means REAL
                pred = 1 if prob > threshold else 0
                label_text = "FAKE" if pred == 1 else "REAL"

                return {
                    "prediction": label_text,
                    "pred_class": pred,
                    "probability": prob,
                    "threshold": threshold,
                    "frames_used": len(frames)
                }
        except Exception as e:
            return {
                "prediction": "ERROR",
                "error": str(e)
            }

# Module-level singleton predictor (so multiple calls reuse model)
_predictor_singleton = None
def get_predictor(model_path=None, threshold=None):
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = SafeVideoPredictor(model_path or DEFAULT_MODEL_PATH, threshold or DEFAULT_THRESHOLD)
    return _predictor_singleton

def predict_video_improved(video_path, model_path=None, threshold=None):
    predictor = get_predictor(model_path, threshold)
    return predictor.predict_video(video_path, threshold)

# If run directly, allow passing a video path
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict deepfake on a single video")
    parser.add_argument("video", help="path to video file")
    parser.add_argument("--model", help="path to checkpoint", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--threshold", help="probability threshold for FAKE", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    res = predict_video_improved(args.video, model_path=args.model, threshold=args.threshold)
    print("Result:", res)
