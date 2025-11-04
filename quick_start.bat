@echo off
REM Quick Start Script for Windows
REM Usage: quick_start.bat [command]

if "%1"=="test" (
    echo 🧪 Running quick test...
    python quick_test_enhanced.py
) else if "%1"=="status" (
    echo 📊 Checking model status...
    python check_model_status.py
) else if "%1"=="monitor" (
    echo 👀 Starting live monitoring...
    python monitor_training.py
) else if "%1"=="train1" (
    echo 🚀 Starting Phase 1 training...
    python models/enhanced_trainer.py
) else if "%1"=="train2" (
    echo 🚀 Starting Phase 2 training...
    python models/phase2_trainer.py
) else if "%1"=="compare" (
    echo 📈 Comparing models...
    python compare_all_models.py
) else if "%1"=="batch" (
    echo 📦 Batch testing videos...
    python batch_test_models.py %2
) else (
    echo 🚀 Quick Start Commands:
    echo.
    echo   test      - Run quick validation test
    echo   status    - Check model training status
    echo   monitor   - Live training monitor
    echo   train1    - Start Phase 1 training
    echo   train2    - Start Phase 2 training
    echo   compare   - Compare all models
    echo   batch     - Batch test videos [directory]
    echo.
    echo Example: quick_start.bat test
)


