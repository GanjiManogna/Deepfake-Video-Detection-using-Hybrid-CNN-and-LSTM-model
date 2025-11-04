"""
Training utilities for deepfake detection models
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Enhanced training loop with gradient clipping and mixed precision"""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
    
    for batch_idx, (frames, labels) in enumerate(train_loader):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if config.USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                predictions, _ = model(frames)
                loss = criterion(predictions, labels.float())
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions, _ = model(frames)
            loss = criterion(predictions, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validation with detailed metrics"""
    model.eval()
    total_loss = 0
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            predictions, _ = model(frames)
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()
            
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    predictions_binary = (np.array(predictions_list) > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, predictions_binary, average='binary')
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics