#!/bin/bash
# Quick Start Script for Model Improvements
# Usage: bash quick_start.sh [command]

case "$1" in
    test)
        echo "🧪 Running quick test..."
        python quick_test_enhanced.py
        ;;
    status)
        echo "📊 Checking model status..."
        python check_model_status.py
        ;;
    monitor)
        echo "👀 Starting live monitoring..."
        python monitor_training.py
        ;;
    train1)
        echo "🚀 Starting Phase 1 training..."
        python models/enhanced_trainer.py
        ;;
    train2)
        echo "🚀 Starting Phase 2 training..."
        python models/phase2_trainer.py
        ;;
    compare)
        echo "📈 Comparing models..."
        python compare_all_models.py
        ;;
    batch)
        echo "📦 Batch testing videos..."
        python batch_test_models.py "${2:-testing_data}"
        ;;
    *)
        echo "🚀 Quick Start Commands:"
        echo ""
        echo "  test      - Run quick validation test"
        echo "  status    - Check model training status"
        echo "  monitor   - Live training monitor"
        echo "  train1    - Start Phase 1 training"
        echo "  train2    - Start Phase 2 training"
        echo "  compare   - Compare all models"
        echo "  batch     - Batch test videos [directory]"
        echo ""
        echo "Example: bash quick_start.sh test"
        ;;
esac


