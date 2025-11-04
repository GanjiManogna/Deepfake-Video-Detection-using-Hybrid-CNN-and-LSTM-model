#!/usr/bin/env python3
"""Quick check if Phase 1 training is running and making progress"""

from pathlib import Path
from datetime import datetime

def quick_check():
    """Quick check training status"""
    saved_models = Path("saved_models")
    
    print("🔍 Quick Training Check")
    print("=" * 40)
    
    # Check for Phase 1 checkpoints
    last_checkpoint = saved_models / "last_enhanced_model.pth"
    best_checkpoint = saved_models / "best_enhanced_model.pth"
    
    if last_checkpoint.exists():
        size = last_checkpoint.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(last_checkpoint.stat().st_mtime)
        age_seconds = (datetime.now() - mtime).total_seconds()
        age_min = age_seconds / 60
        
        print(f"✅ Training in progress!")
        print(f"   Last checkpoint: {age_min:.1f} min ago")
        print(f"   Size: {size:.2f} MB")
        
        if age_min < 5:
            print(f"   Status: 🟢 Active (recent checkpoint)")
        elif age_min < 30:
            print(f"   Status: 🟡 May be training epoch")
        else:
            print(f"   Status: 🔴 No recent activity")
    else:
        print("⏳ Training starting up...")
        print("   (Checkpoint will appear in 1-5 minutes)")
        print("   Data loading takes time initially")
    
    if best_checkpoint.exists():
        size = best_checkpoint.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(best_checkpoint.stat().st_mtime)
        print(f"\n✅ Best model found!")
        print(f"   Size: {size:.2f} MB")
        print(f"   Saved: {mtime.strftime('%H:%M:%S')}")
    
    print("\n💡 Run again in a few minutes to see progress")

if __name__ == "__main__":
    quick_check()


