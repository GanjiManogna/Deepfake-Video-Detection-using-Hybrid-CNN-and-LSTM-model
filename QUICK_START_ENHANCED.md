# Quick Start: Enhanced Model Training

## 🚀 Quick Test (Recommended First)

Before full training, verify everything works:

```bash
python quick_test_enhanced.py
```

This will:
- ✅ Test dataset loading
- ✅ Test model forward pass
- ✅ Test training step
- Only uses 10 videos and 2 epochs (very fast)

## 📊 Train Enhanced Model

```bash
python models/enhanced_trainer.py
```

### What It Does:
- Uses **enhanced data augmentation** (brightness, contrast, blur, rotation, color jitter)
- Trains with **24 frames** (vs 16) for better temporal context
- Uses **Focal Loss** for better handling of hard examples
- Applies **Test-Time Augmentation** during validation
- Tracks **F1-score** (better metric than just accuracy)
- Saves as `saved_models/best_enhanced_model.pth`

### Expected Time:
- ~2-4 hours on GPU (depending on dataset size)
- ~8-16 hours on CPU

### Expected Improvements:
- **+3-5% accuracy** over original model
- Better generalization to real-world videos
- More stable predictions

## 🔍 Compare Results

After training, compare performance:

```bash
python compare_enhanced_model.py
```

This will:
- Load both original and enhanced models
- Evaluate on same test set
- Show detailed comparison metrics
- Save results to `training_results/model_comparison.json`

## 🔄 Use Enhanced Model in Web App

To use the enhanced model in your web interface:

1. **Option 1: Direct Replacement** (Recommended)
   ```python
   # In web/improved_predictor.py, change:
   DEFAULT_MODEL_PATH = Path(...) / "saved_models" / "best_enhanced_model.pth"
   ```

2. **Option 2: Keep Both Models**
   ```python
   # Add environment variable support:
   USE_ENHANCED = os.getenv("USE_ENHANCED_MODEL", "0") == "1"
   model_path = "best_enhanced_model.pth" if USE_ENHANCED else "best_simple_model.pth"
   ```

## 📈 Monitoring Training

Training will show:
- Train/Val loss and accuracy
- F1-score (key metric)
- Learning rate schedule
- Confusion matrix
- Best model saved automatically

## ⚙️ Configuration

Edit `models/enhanced_trainer.py` to adjust:

- `SEQ_LEN`: 24 (increase for longer sequences)
- `AUG_BRIGHTNESS`: 0.15 (brightness variation)
- `FOCAL_LOSS_GAMMA`: 2.0 (hard example focus)
- `LABEL_SMOOTHING`: 0.1 (regularization)
- `USE_TTA`: True (test-time augmentation)

## 🎯 Expected Results

Based on typical deepfake detection benchmarks:

| Metric | Original | Enhanced | Improvement |
|--------|----------|----------|------------|
| Accuracy | ~85-90% | ~88-93% | +3-5% |
| F1-Score | ~0.82-0.88 | ~0.85-0.91 | +0.03-0.05 |
| Precision | ~0.80-0.85 | ~0.83-0.88 | +3-5% |
| Recall | ~0.85-0.90 | ~0.87-0.92 | +2-4% |

*Note: Actual results depend on your dataset and test conditions*

## 🔧 Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` from 2 to 1
- Reduce `SEQ_LEN` from 24 to 16
- Use mixed precision training

### Training Too Slow
- Reduce dataset size: `max_videos_per_dataset=100`
- Reduce `EPOCHS` for testing
- Ensure GPU acceleration is working

### Poor Performance
- Check data balance (real vs fake ratio)
- Verify data augmentation is working
- Try different learning rates
- Increase training epochs

## 📚 Next Steps

After Phase 1 (Enhanced Trainer):
1. Evaluate on your test videos
2. If good, proceed to Phase 2 (architecture improvements)
3. See `MODEL_IMPROVEMENTS.md` for full roadmap

## 💡 Tips

- **Always validate**: Use a held-out test set
- **Monitor overfitting**: Watch train vs val metrics
- **Save checkpoints**: Best model is auto-saved
- **Experiment**: Try different augmentation strengths
- **Iterate**: Training is an iterative process

---

**Questions?** Check `MODEL_IMPROVEMENTS.md` for detailed explanations of each improvement.


