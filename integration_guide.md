# Integration Guide: Using Advanced Features

## 🚀 Quick Integration Steps

### 1. Use Advanced Ensemble Fusion (Available Now!)

The advanced ensemble is already integrated into `web/simple_tri_predictor.py`. It will automatically use weighted confidence fusion if available.

**How it works:**
- If `web/advanced_ensemble.py` exists, it uses advanced fusion
- Falls back to simple majority vote if not available
- Set `ENSEMBLE_STRATEGY` environment variable to change strategy

**Strategies:**
```bash
# Weighted by confidence (default, recommended)
export ENSEMBLE_STRATEGY=weighted_confidence

# Simple majority vote
export ENSEMBLE_STRATEGY=majority

# Average probabilities in logit space
export ENSEMBLE_STRATEGY=probability_avg

# Use learned weights
export ENSEMBLE_STRATEGY=learned_weights
```

### 2. Use Enhanced Model (After Training)

Once Phase 1 training completes:

```bash
# Set environment variable
export USE_ENHANCED_MODEL=1

# Or on Windows PowerShell:
$env:USE_ENHANCED_MODEL="1"

# Restart your Flask app
python app.py
```

The web app will automatically use `best_enhanced_model.pth` instead of `best_simple_model.pth`.

### 3. Use Phase 2 Model (After Training)

To use the Phase 2 model (most advanced):

1. **Train Phase 2:**
   ```bash
   python models/phase2_trainer.py
   ```

2. **Update predictor manually:**
   Edit `web/improved_predictor.py`:
   ```python
   DEFAULT_MODEL_PATH = Path(...) / "saved_models" / "best_phase2_model.pth"
   ```

## 📊 Monitoring Training

### Quick Status Check
```bash
python check_model_status.py
```

### Live Monitoring
```bash
python monitor_training.py
# Or with custom interval (30 seconds):
python monitor_training.py 30
```

### Start Training (if not running)
```bash
python start_phase1_training.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `USE_ENHANCED_MODEL` | `0` (default), `1` | Use Phase 1 enhanced model |
| `ENSEMBLE_STRATEGY` | `weighted_confidence`, `majority`, `probability_avg`, `learned_weights` | Ensemble fusion method |
| `HYBRID_SEQ_LEN` | `16`, `24`, `32` | Sequence length for hybrid model |
| `LSTM_TEMPERATURE` | `0.5-2.0` | Confidence calibration for LSTM |
| `PRED_THRESHOLD` | `0.4-0.6` | Prediction threshold |

### Example Configuration

```bash
# Use enhanced model with advanced ensemble
export USE_ENHANCED_MODEL=1
export ENSEMBLE_STRATEGY=weighted_confidence
export HYBRID_SEQ_LEN=24

# Start Flask app
python app.py
```

## 📈 Performance Comparison

After training completes, compare models:

```bash
python compare_enhanced_model.py
```

This will show:
- Accuracy comparison
- Precision/Recall/F1 scores
- Confusion matrices
- Best performing model

## 🎯 Recommended Workflow

1. **Phase 1 Training** (2-4 hours)
   - Run: `python models/enhanced_trainer.py`
   - Monitor: `python monitor_training.py`
   - Expected: +3-5% accuracy

2. **Evaluate Phase 1**
   - Run: `python compare_enhanced_model.py`
   - Check if improvement is significant
   - If good, proceed to Phase 2

3. **Phase 2 Training** (3-5 hours, optional)
   - Run: `python models/phase2_trainer.py`
   - Monitor progress
   - Expected: +6-10% over Phase 1

4. **Integration**
   - Use best performing model
   - Enable advanced ensemble fusion
   - Test on real-world videos

## 🔍 Troubleshooting

### Training Not Starting
- Check data paths: Videos should be in `data/faceforensics_data/`
- Verify data structure: Should have `c23/videos/` subdirectories
- Check logs for errors

### Model Not Loading
- Verify checkpoint exists in `saved_models/`
- Check model architecture matches checkpoint
- Try loading with `strict=False` (already implemented)

### Ensemble Not Working
- Verify `web/advanced_ensemble.py` exists
- Check import errors in console
- Falls back to majority vote automatically

### Low Accuracy
- Check data quality and balance
- Adjust threshold: `PRED_THRESHOLD=0.5`
- Try different ensemble strategy
- Consider longer training (more epochs)

## 💡 Tips

1. **Start Small**: Test with Phase 1 before Phase 2
2. **Monitor Progress**: Use monitoring tools to track training
3. **Compare Models**: Always compare before switching
4. **Keep Backups**: Don't delete old models until new ones are proven
5. **Test Thoroughly**: Test on diverse real-world videos

## 📚 Additional Resources

- `MODEL_IMPROVEMENTS.md` - Full improvement roadmap
- `QUICK_START_ENHANCED.md` - Phase 1 training guide
- `PHASE2_IMPLEMENTATION.md` - Phase 2 architecture details
- `IMPLEMENTATION_SUMMARY.md` - Complete overview


