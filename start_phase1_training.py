#!/usr/bin/env python3
"""
Start Phase 1 training if not already running
"""

import sys
import subprocess
from pathlib import Path

def is_training_running():
    """Check if training process is already running"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any('enhanced_trainer' in str(arg) for arg in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    except ImportError:
        # psutil not available, check file modification times instead
        last_checkpoint = Path("saved_models/last_enhanced_model.pth")
        if last_checkpoint.exists():
            import time
            age = time.time() - last_checkpoint.stat().st_mtime
            # If checkpoint was modified in last 5 minutes, assume training is active
            return age < 300
        return False

def start_training():
    """Start Phase 1 training"""
    trainer_path = Path("models/enhanced_trainer.py")
    
    if not trainer_path.exists():
        print(f"❌ Trainer not found: {trainer_path}")
        return False
    
    if is_training_running():
        print("✅ Training already running (detected active process or recent checkpoint)")
        return True
    
    print("🚀 Starting Phase 1 Enhanced Training...")
    print("   This will run in the background.")
    print("   Monitor with: python monitor_training.py")
    print("   Check status with: python check_model_status.py")
    print()
    
    # Start training
    try:
        # On Windows, use START command to run in new window
        if sys.platform == 'win32':
            subprocess.Popen([
                sys.executable, str(trainer_path)
            ], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # On Unix, use nohup or run in background
            subprocess.Popen([
                sys.executable, str(trainer_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ Training process started!")
        print("   Check 'saved_models/' directory for checkpoints")
        return True
    except Exception as e:
        print(f"❌ Failed to start training: {e}")
        return False

if __name__ == "__main__":
    start_training()


