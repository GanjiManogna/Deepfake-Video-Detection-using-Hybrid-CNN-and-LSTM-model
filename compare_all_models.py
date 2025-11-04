#!/usr/bin/env python3
"""
Compare all available models (Original, Phase 1, Phase 2)
on a test set and generate comprehensive report
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def load_model_results(model_path, test_loader, device, model_class, config_class):
    """Load and evaluate a model"""
    try:
        model = model_class(config_class()).to(device)
        
        if not model_path.exists():
            return None
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().long().numpy())
        
        if not all_preds:
            return None
        
        return {
            'predictions': all_preds,
            'targets': all_targets,
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds).tolist()
        }
    except Exception as e:
        print(f"  ❌ Error loading {model_path.name}: {e}")
        return None

def compare_all_models():
    """Compare all available models"""
    print("📊 Comprehensive Model Comparison")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Import required classes
    try:
        from models.simple_improved_trainer import SimpleConfig, SimpleDeepfakeModel, SimpleTrainer, SimpleVideoDataset
        from models.enhanced_trainer import EnhancedConfig, EnhancedTrainer
        from models.phase2_improvements import Phase2Config, Phase2DeepfakeModel, Phase2Trainer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Prepare test dataset
    print("📁 Preparing test dataset...")
    trainer = SimpleTrainer(SimpleConfig())
    
    try:
        # Get all data
        ff_paths, ff_labels = trainer.load_faceforensics_data(200)
        dfdc_paths, dfdc_labels = trainer.load_dfdc_data(200)
        
        all_paths = ff_paths + dfdc_paths
        all_labels = ff_labels + dfdc_labels
        
        if len(all_paths) == 0:
            print("⚠️  No test data available. Using dummy test...")
            return None
        
        # Split: 80% train/val, 20% test
        from sklearn.model_selection import train_test_split
        _, test_paths, _, test_labels = train_test_split(
            all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        print(f"  Test set: {len(test_paths)} videos")
        print(f"  Real: {test_labels.count(0)}, Fake: {test_labels.count(1)}")
        
        # Create test dataset
        _, val_t = trainer.create_transforms()
        test_dataset = SimpleVideoDataset(
            test_paths, test_labels,
            transform=val_t,
            seq_len=SimpleConfig().SEQ_LEN,
            frame_size=SimpleConfig().FRAME_SIZE
        )
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
        
    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        return None
    
    # Test all models
    saved_models = Path("saved_models")
    results = {}
    
    models_to_test = {
        "Original": {
            "path": saved_models / "best_simple_model.pth",
            "config": SimpleConfig,
            "model": SimpleDeepfakeModel
        },
        "Phase 1 Enhanced": {
            "path": saved_models / "best_enhanced_model.pth",
            "config": EnhancedConfig,
            "model": SimpleDeepfakeModel  # Same architecture, better training
        },
        "Phase 2 Advanced": {
            "path": saved_models / "best_phase2_model.pth",
            "config": Phase2Config,
            "model": Phase2DeepfakeModel
        }
    }
    
    print("\n🔍 Testing models...")
    print("-" * 60)
    
    for model_name, model_info in models_to_test.items():
        print(f"\n{model_name}:")
        if not model_info["path"].exists():
            print(f"  ⚠️  Not found: {model_info['path'].name}")
            continue
        
        result = load_model_results(
            model_info["path"],
            test_loader,
            device,
            model_info["model"],
            model_info["config"]
        )
        
        if result:
            results[model_name] = result
            print(f"  ✅ Accuracy: {result['accuracy']:.4f}")
            print(f"     Precision: {result['precision']:.4f}")
            print(f"     Recall: {result['recall']:.4f}")
            print(f"     F1-Score: {result['f1']:.4f}")
    
    # Comparison summary
    if results:
        print("\n" + "=" * 60)
        print("📈 Comparison Summary")
        print("=" * 60)
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        baseline_f1 = None
        for model_name, result in results.items():
            if baseline_f1 is None:
                baseline_f1 = result['f1']
            
            improvement = ((result['f1'] - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            marker = "🏆" if result['f1'] == max(r['f1'] for r in results.values()) else "  "
            
            print(f"{marker} {model_name:<18} {result['accuracy']:<12.4f} "
                  f"{result['precision']:<12.4f} {result['recall']:<12.4f} {result['f1']:<12.4f}")
            if improvement != 0:
                print(f"    Improvement: {improvement:+.2f}%")
        
        # Save results
        output_file = Path("testing_results/comprehensive_comparison.json")
        output_file.parent.mkdir(exist_ok=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_set_size": len(test_paths),
            "results": {
                name: {
                    "accuracy": float(result["accuracy"]),
                    "precision": float(result["precision"]),
                    "recall": float(result["recall"]),
                    "f1": float(result["f1"]),
                    "confusion_matrix": result["confusion_matrix"]
                }
                for name, result in results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Detailed report saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    compare_all_models()


