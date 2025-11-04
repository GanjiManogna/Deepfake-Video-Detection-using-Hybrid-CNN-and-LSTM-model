#!/usr/bin/env python3
"""
Quick check of model training status
"""

from pathlib import Path
from datetime import datetime

def check_model_status():
    """Check status of all models"""
    saved_models = Path("saved_models")
    
    print("📊 Model Training Status")
    print("=" * 60)
    
    models = {
        "Original (Simple)": ["best_simple_model.pth", "last_simple_model.pth"],
        "Phase 1 (Enhanced)": ["best_enhanced_model.pth", "last_enhanced_model.pth"],
        "Phase 2 (Advanced)": ["best_phase2_model.pth", "last_phase2_model.pth"]
    }
    
    for model_name, files in models.items():
        print(f"\n{model_name}:")
        best_file = saved_models / files[0]
        last_file = saved_models / files[1]
        
        if best_file.exists():
            size = best_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(best_file.stat().st_mtime)
            print(f"  ✅ Best model: {size:.2f} MB")
            print(f"     Last saved: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  ❌ Best model: Not found")
        
        if last_file.exists():
            size = last_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(last_file.stat().st_mtime)
            age_min = (datetime.now() - mtime).total_seconds() / 60
            print(f"  🏃 Last checkpoint: {size:.2f} MB")
            print(f"     Modified: {age_min:.1f} min ago")
            if age_min < 10:
                print(f"     Status: 🟢 Training in progress!")
            else:
                print(f"     Status: 🟡 No recent updates")
        else:
            print(f"  ⚠️  Last checkpoint: Not found")
    
    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("  • Run 'python monitor_training.py' for live monitoring")
    print("  • Check training logs for detailed progress")
    print("  • Best model is saved when validation improves")

if __name__ == "__main__":
    check_model_status()


