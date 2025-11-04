#!/usr/bin/env python3
"""
Compare original model vs enhanced model performance
"""

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import models
import sys
sys.path.append(str(Path(__file__).parent / "models"))

from models.simple_improved_trainer import SimpleConfig, SimpleDeepfakeModel, SimpleTrainer
from models.enhanced_trainer import EnhancedConfig, EnhancedTrainer

def evaluate_model(model, test_loader, device, use_tta=False):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            
            if use_tta:
                # Test-time augmentation: flip and average
                sequences_flip = sequences.flip(dims=[4])
                outputs_flip = model(sequences_flip)
                outputs = (outputs + outputs_flip) / 2.0
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().long().numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }

def main():
    """Compare original vs enhanced models"""
    print("🔍 Comparing Original vs Enhanced Models")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Prepare test dataset (same for both)
    from torch.utils.data import DataLoader
    
    # Use original config for test dataset
    orig_config = SimpleConfig()
    orig_trainer = SimpleTrainer(orig_config)
    
    # Get test split
    from sklearn.model_selection import train_test_split
    ff_paths, ff_labels = orig_trainer.load_faceforensics_data(200)
    dfdc_paths, dfdc_labels = orig_trainer.load_dfdc_data(200)
    
    all_paths = ff_paths + dfdc_paths
    all_labels = ff_labels + dfdc_labels
    
    # Split train/val/test: 64% train, 16% val, 20% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.36, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.556, random_state=42, stratify=temp_labels
    )
    
    print(f"Test set: {len(test_paths)} videos")
    print(f"  Real: {test_labels.count(0)}, Fake: {test_labels.count(1)}\n")
    
    # Create test dataset (use validation transforms)
    _, val_t = orig_trainer.create_transforms()
    from models.simple_improved_trainer import SimpleVideoDataset
    
    test_dataset = SimpleVideoDataset(
        test_paths, test_labels,
        transform=val_t,
        seq_len=orig_config.SEQ_LEN,
        frame_size=orig_config.FRAME_SIZE
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    results = {}
    
    # Test Original Model
    print("📊 Testing Original Model...")
    print("-" * 60)
    
    orig_model = SimpleDeepfakeModel(orig_config).to(device)
    orig_path = Path("saved_models/best_simple_model.pth")
    
    if orig_path.exists():
        try:
            orig_model.load_state_dict(torch.load(orig_path, map_location=device))
            print("✅ Loaded original model checkpoint")
            
            orig_results = evaluate_model(orig_model, test_loader, device, use_tta=False)
            results['original'] = orig_results
            
            print(f"  Accuracy:  {orig_results['accuracy']:.4f}")
            print(f"  Precision: {orig_results['precision']:.4f}")
            print(f"  Recall:    {orig_results['recall']:.4f}")
            print(f"  F1-Score:  {orig_results['f1']:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    {orig_results['confusion_matrix']}")
        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            results['original'] = None
    else:
        print(f"⚠️  Original model checkpoint not found: {orig_path}")
        results['original'] = None
    
    print()
    
    # Test Enhanced Model
    print("📊 Testing Enhanced Model...")
    print("-" * 60)
    
    enh_config = EnhancedConfig()
    enh_model = SimpleDeepfakeModel(enh_config).to(device)  # Same architecture
    enh_path = Path("saved_models/best_enhanced_model.pth")
    
    if enh_path.exists():
        try:
            enh_model.load_state_dict(torch.load(enh_path, map_location=device))
            print("✅ Loaded enhanced model checkpoint")
            
            # Test without TTA
            enh_results_no_tta = evaluate_model(enh_model, test_loader, device, use_tta=False)
            print("\n  Without TTA:")
            print(f"    Accuracy:  {enh_results_no_tta['accuracy']:.4f}")
            print(f"    Precision: {enh_results_no_tta['precision']:.4f}")
            print(f"    Recall:    {enh_results_no_tta['recall']:.4f}")
            print(f"    F1-Score:  {enh_results_no_tta['f1']:.4f}")
            
            # Test with TTA
            enh_results_tta = evaluate_model(enh_model, test_loader, device, use_tta=True)
            print("\n  With TTA:")
            print(f"    Accuracy:  {enh_results_tta['accuracy']:.4f}")
            print(f"    Precision: {enh_results_tta['precision']:.4f}")
            print(f"    Recall:    {enh_results_tta['recall']:.4f}")
            print(f"    F1-Score:  {enh_results_tta['f1']:.4f}")
            
            results['enhanced_no_tta'] = enh_results_no_tta
            results['enhanced_tta'] = enh_results_tta
        except Exception as e:
            print(f"❌ Failed to load enhanced model: {e}")
            results['enhanced_no_tta'] = None
            results['enhanced_tta'] = None
    else:
        print(f"⚠️  Enhanced model checkpoint not found: {enh_path}")
        print("   Run 'python models/enhanced_trainer.py' to train it first!")
        results['enhanced_no_tta'] = None
        results['enhanced_tta'] = None
    
    print()
    
    # Comparison
    if results['original'] and results['enhanced_tta']:
        print("📈 Comparison Summary")
        print("=" * 60)
        
        orig_f1 = results['original']['f1']
        enh_f1 = results['enhanced_tta']['f1']
        improvement = enh_f1 - orig_f1
        
        print(f"\nOriginal Model F1: {orig_f1:.4f}")
        print(f"Enhanced Model F1: {enh_f1:.4f}")
        print(f"Improvement:       {improvement:+.4f} ({improvement/orig_f1*100:+.2f}%)")
        
        print(f"\nDetailed Metrics:")
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        print(f"{'Metric':<12} {'Original':<12} {'Enhanced':<12} {'Change':<12}")
        print("-" * 50)
        for m in metrics:
            orig_val = results['original'][m]
            enh_val = results['enhanced_tta'][m]
            change = enh_val - orig_val
            print(f"{m.capitalize():<12} {orig_val:<12.4f} {enh_val:<12.4f} {change:+.4f}")
    
    # Save results
    results_path = Path("training_results/model_comparison.json")
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, val in results.items():
        if val is not None:
            json_results[key] = val
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")

if __name__ == "__main__":
    main()


