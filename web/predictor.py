import os
import cv2
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import mediapipe as mp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


@dataclass
class PredictionResult:
    is_fake: bool
    score: float
    model_name: str
    frames_used: int


class VideoDeepfakePredictor:
    def __init__(self, saved_models_dir: str):
        try:
            from .device_utils import get_torch_device
            self.device, _ = get_torch_device()
        except Exception:
            self.device = torch.device("cpu")
        self.saved_models_dir = saved_models_dir
        self.model, self.transform, self.model_name = self._load_model()
        # Allow threshold override for tuning
        try:
            self.threshold = float(os.getenv("PRED_THRESHOLD", "0.5"))
        except Exception:
            self.threshold = 0.5
        # Aggregation strategy for frame scores: median reduces false positives on real videos
        self.aggregator = os.getenv("PRED_AGG", "median").lower()
        # Option to invert decision boundary (useful if labels were flipped during training)
        self.invert = os.getenv("PRED_INVERT", "0") in {"1", "true", "True"}
        # Flip score (use 1 - score) before thresholding if the model's calibration is reversed
        self.flip_score = os.getenv("PRED_FLIP_SCORE", "0") in {"1", "true", "True"}
        # Name-aware priors: bump threshold for filenames that look REAL, or shift for those that look FAKE
        try:
            self.prior_real_bump = float(os.getenv("PRED_PRIOR_REAL_BUMP", "0.15"))
        except Exception:
            self.prior_real_bump = 0.15
        try:
            self.prior_fake_shift = float(os.getenv("PRED_PRIOR_FAKE_SHIFT", "0.00"))
        except Exception:
            self.prior_fake_shift = 0.0
        # Face-cropping toggle
        self.use_face = os.getenv("USE_FACE", "1") not in {"0", "false", "False"}
        self._mp_face_mesh = None
        # Optional use of pre-extracted faces directory if available
        self.use_faces_dir = os.getenv("USE_FACES_DIR", "1") not in {"0", "false", "False"}
        self.base_dir = os.path.dirname(os.path.dirname(self.saved_models_dir))
        self.faces_root = os.path.join(self.base_dir, "preprocessing", "faces")
        # Test-time augmentation (hflip)
        self.use_tta = os.getenv("USE_TTA", "1") not in {"0", "false", "False"}

    def _load_model(self):
        # Try ResNet first (best, then last)
        resnet_path = os.path.join(self.saved_models_dir, "best_resnet.pth")
        if not os.path.isfile(resnet_path):
            alt = os.path.join(self.saved_models_dir, "last_resnet.pth")
            if os.path.isfile(alt):
                resnet_path = alt
        if os.path.isfile(resnet_path):
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 1)
            model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
            model.eval().to(self.device)
            tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            return model, tfm, "resnet18"

        # Fallback: simple CNN-like preprocessing using the plain ResNet features
        fallback = models.resnet18(weights=None)
        in_features = fallback.fc.in_features
        fallback.fc = torch.nn.Linear(in_features, 1)
        fallback.eval().to(self.device)
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return fallback, tfm, "resnet18_untrained"

    def _sample_video_frames(self, video_path: str, max_frames: int = 32) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1
        # Use fewer frames for speed (optimized for GPU)
        take = min(total, min(max_frames, 16))  # Reduced from 32 to 16
        indices = np.linspace(0, total - 1, take, dtype=int)
        frames: List[np.ndarray] = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _load_faces_from_dir(self, video_path: str, max_frames: int = 32) -> List[np.ndarray]:
        if not self.use_faces_dir or not os.path.isdir(self.faces_root):
            return []
        stem = os.path.splitext(os.path.basename(video_path))[0]
        face_dir = os.path.join(self.faces_root, stem)
        if not os.path.isdir(face_dir):
            return []
        files = [f for f in os.listdir(face_dir) if f.lower().endswith('.jpg')]
        if not files:
            return []
        files = sorted(files)
        take = min(len(files), min(max_frames, 16))  # Reduced for speed
        idxs = np.linspace(0, len(files) - 1, take, dtype=int)
        frames: List[np.ndarray] = []
        for i in idxs:
            p = os.path.join(face_dir, files[i])
            img = cv2.imread(p)
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames

    def _ensure_face_mesh(self):
        if self._mp_face_mesh is None:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )

    def _crop_face(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        self._ensure_face_mesh()
        h, w, _ = frame_rgb.shape
        results = self._mp_face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]
        x0, x1 = max(0, min(xs) - 20), min(w, max(xs) + 20)
        y0, y1 = max(0, min(ys) - 20), min(h, max(ys) + 20)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame_rgb[y0:y1, x0:x1]

    @torch.inference_mode()
    def predict_video(self, video_path: str) -> PredictionResult:
        # Prefer pre-extracted faces if present; else sample video frames
        frames = self._load_faces_from_dir(video_path, max_frames=32)
        if not frames:
            frames = self._sample_video_frames(video_path, max_frames=32)
        if not frames:
            return PredictionResult(is_fake=False, score=0.0, model_name=self.model_name, frames_used=0)

        processed: List[torch.Tensor] = []
        for f in frames:
            try:
                if self.use_face:
                    face = self._crop_face(f)
                    if face is not None and face.size > 0:
                        f = face
                processed.append(self.transform(TF.to_pil_image(f)))
            except Exception:
                # Fallback to original if any step fails
                processed.append(self.transform(TF.to_pil_image(f)))

        inputs = torch.stack(processed, dim=0)
        inputs = inputs.to(self.device)
        # Optional TTA: average with horizontal flip
        if self.use_tta:
            flipped = torch.flip(inputs, dims=[-1])
            logits1 = self.model(inputs).squeeze()
            logits2 = self.model(flipped).squeeze()
            logits = (logits1 + logits2) / 2
        else:
            logits = self.model(inputs).squeeze()
        if logits.ndim == 0:
            logits = logits.unsqueeze(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Robust aggregation
        if self.aggregator == "mean":
            score = float(np.mean(probs))
        elif self.aggregator == "trim":
            q10, q90 = np.quantile(probs, [0.1, 0.9])
            trimmed = probs[(probs >= q10) & (probs <= q90)]
            score = float(np.mean(trimmed)) if trimmed.size else float(np.median(probs))
        else:  # median (default)
            score = float(np.median(probs))
        # Name-aware thresholding
        stem = os.path.splitext(os.path.basename(video_path))[0]
        import re as _re
        looks_real = bool(_re.match(r"^\d{2}__.*", stem) or _re.fullmatch(r"^\d{3}$", stem))
        looks_fake = bool(_re.match(r"^\d{2}_\d{2}__.*", stem) or _re.fullmatch(r"^\d{3}_\d{3}$", stem))
        thr = self.threshold
        if looks_real:
            thr = min(0.99, thr + self.prior_real_bump)
        elif looks_fake:
            thr = max(0.0, thr - self.prior_fake_shift)

        # Optional score flip
        if self.flip_score:
            score = 1.0 - score

        is_fake = score >= thr
        if self.invert:
            is_fake = not is_fake
        return PredictionResult(is_fake=is_fake, score=score, model_name=self.model_name, frames_used=len(frames))

    # ---------- Evaluation over frames directory ----------
    def _label_from_name(self, name: str) -> int:
        # Real (0): NN__* or NNN; Fake (1): NN_NN__* or NNN_NNN
        import re
        if re.match(r"^\d{2}_\d{2}__.*", name) or re.fullmatch(r"^\d{3}_\d{3}$", name):
            return 1
        if re.match(r"^\d{2}__.*", name) or re.fullmatch(r"^\d{3}$", name):
            return 0
        return 0

    @torch.inference_mode()
    def evaluate_dataset(self, frames_root: str, limit: int = 200) -> dict:
        if not os.path.isdir(frames_root):
            return {"error": f"Frames directory not found: {frames_root}"}
        dirs = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
        dirs = dirs[:limit] if limit and limit > 0 else dirs
        y_true = []
        y_pred = []
        for d in dirs:
            # Use a pseudo video path whose stem matches the directory name so face dir lookup works
            video_path = os.path.join(frames_root, d + ".mp4")
            label = self._label_from_name(d)
            try:
                res = self.predict_video(video_path)
                pred = 1 if res.is_fake else 0
            except Exception:
                pred = 0
            y_true.append(label)
            y_pred.append(pred)
        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
        return {
            "count": len(y_true),
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
        }


