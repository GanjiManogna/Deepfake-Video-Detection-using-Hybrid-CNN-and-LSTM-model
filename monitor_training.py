#!/usr/bin/env python3
"""
Monitor training progress by checking checkpoint files and logs
"""

import os
import time
from pathlib import Path
from datetime import datetime

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0

def get_latest_checkpoint(checkpoint_dir, pattern="best_*.pth"):
    """Get the latest checkpoint file"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]

def monitor_training(checkpoint_dir="saved_models", interval=60):
    """Monitor training progress"""
    checkpoint_dir = Path(checkpoint_dir)
    
    print("🔍 Training Monitor")
    print("=" * 60)
    print(f"Monitoring: {checkpoint_dir.absolute()}")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    last_sizes = {}
    
    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] Checking checkpoints...")
            
            # Check for different model types
            models_to_check = {
                "Phase 1 Enhanced": "best_enhanced_model.pth",
                "Phase 2": "best_phase2_model.pth",
                "Original": "best_simple_model.pth"
            }
            
            found_any = False
            for model_name, pattern in models_to_check.items():
                checkpoint = get_latest_checkpoint(checkpoint_dir, pattern)
                if checkpoint:
                    size_mb = get_file_size_mb(checkpoint)
                    mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime)
                    age = datetime.now() - mtime
                    
                    # Check if file was updated
                    was_updated = checkpoint not in last_sizes or last_sizes[checkpoint] != size_mb
                    
                    status = "✅ NEW!" if was_updated else "✅"
                    print(f"  {status} {model_name}:")
                    print(f"    File: {checkpoint.name}")
                    print(f"    Size: {size_mb:.2f} MB")
                    print(f"    Last modified: {mtime.strftime('%H:%M:%S')} ({age.total_seconds()/60:.1f} min ago)")
                    
                    last_sizes[checkpoint] = size_mb
                    found_any = True
            
            if not found_any:
                # Check for "last" checkpoints (indicates training in progress)
                last_checkpoints = {
                    "Phase 1 (in progress)": "last_enhanced_model.pth",
                    "Phase 2 (in progress)": "last_phase2_model.pth"
                }
                
                for model_name, pattern in last_checkpoints.items():
                    checkpoint = checkpoint_dir / pattern
                    if checkpoint.exists():
                        size_mb = get_file_size_mb(checkpoint)
                        mtime = datetime.fromtimestamp(checkpoint.stat().st_mtime)
                        age = datetime.now() - mtime
                        
                        print(f"  🏃 {model_name}:")
                        print(f"    File: {checkpoint.name}")
                        print(f"    Size: {size_mb:.2f} MB")
                        print(f"    Last modified: {mtime.strftime('%H:%M:%S')} ({age.total_seconds()/60:.1f} min ago)")
                        print(f"    Status: Training in progress...")
                        found_any = True
                
                if not found_any:
                    print("  ⚠️  No training checkpoints found yet")
                    print("     Waiting for training to start...")
            
            print("\n" + "-" * 60)
            print(f"Next check in {interval} seconds... (Ctrl+C to stop)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for custom interval
    interval = 60
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print("Usage: python monitor_training.py [interval_seconds]")
            sys.exit(1)
    
    monitor_training(interval=interval)


