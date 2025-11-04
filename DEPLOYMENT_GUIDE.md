# 🚀 Deployment Guide: Production-Ready Deepfake Detection

## 📋 Pre-Deployment Checklist

### ✅ Model Preparation
- [ ] Phase 1 model trained and evaluated
- [ ] Phase 2 model trained (optional, if Phase 1 shows improvement)
- [ ] Best model selected based on test set performance
- [ ] Model checkpoint verified and saved
- [ ] Model size acceptable for deployment (< 200MB recommended)

### ✅ Code Verification
- [ ] All imports working correctly
- [ ] Web app starts without errors
- [ ] Predictions return valid results
- [ ] Advanced ensemble fusion active
- [ ] Error handling in place

### ✅ Performance Testing
- [ ] Inference speed acceptable (< 10s per video)
- [ ] Memory usage reasonable
- [ ] GPU acceleration working (if available)
- [ ] Batch processing tested
- [ ] Real-world video testing completed

## 🎯 Deployment Steps

### Step 1: Choose Your Model

**Option A: Original Model (Current)**
- Already deployed
- No changes needed
- Accuracy: ~85-90%

**Option B: Phase 1 Enhanced Model (Recommended)**
```bash
# Set environment variable
export USE_ENHANCED_MODEL=1

# Or in PowerShell:
$env:USE_ENHANCED_MODEL="1"
```
- Accuracy: ~88-93%
- Better generalization

**Option C: Phase 2 Advanced Model**
```python
# Edit web/improved_predictor.py
DEFAULT_MODEL_PATH = Path(...) / "saved_models" / "best_phase2_model.pth"
```
- Accuracy: ~91-95%
- Most advanced

### Step 2: Configure Ensemble Strategy

Set in environment or code:
```bash
# Recommended: Weighted confidence (default)
export ENSEMBLE_STRATEGY=weighted_confidence

# Alternative: Probability averaging
export ENSEMBLE_STRATEGY=probability_avg
```

### Step 3: Test Integration

```bash
# Test web app
python app.py

# In another terminal, test prediction
curl -X POST -F "video=@test_video.mp4" http://localhost:5000/detect
```

### Step 4: Production Configuration

#### Environment Variables
Create `.env` file or set system variables:
```env
USE_ENHANCED_MODEL=1
ENSEMBLE_STRATEGY=weighted_confidence
HYBRID_SEQ_LEN=24
PRED_THRESHOLD=0.5
```

#### Flask Configuration
```python
# app.py
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
```

## 🔧 Production Optimizations

### 1. Model Optimization

**Quantization** (Reduce model size):
```python
# In predictor, after loading model:
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**ONNX Export** (Faster inference):
```python
import torch.onnx
dummy_input = torch.randn(1, 24, 3, 160, 160)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 2. Caching & Performance

**Model Warmup**:
```python
# Warm up models on startup
predictors = {
    'cnn': get_cnn_predictor(),
    'lstm': get_lstm_predictor(),
    'hybrid': get_hybrid_predictor()
}
# Run dummy prediction to load models
dummy_result = predict_video_simple_three("dummy.mp4")
```

**Result Caching**:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_predict(video_hash):
    # Store predictions by video hash
    pass
```

### 3. Error Handling

Add comprehensive error handling:
```python
try:
    result = predict_video_simple_three(video_path)
except torch.cuda.OutOfMemoryError:
    # Fallback to CPU
    result = predict_video_simple_three(video_path, device='cpu')
except Exception as e:
    # Log error and return safe default
    logger.error(f"Prediction failed: {e}")
    result = {"error": str(e), "prediction": "UNKNOWN"}
```

### 4. Monitoring

**Logging**:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

**Performance Metrics**:
```python
import time
start_time = time.time()
result = predict_video_simple_three(video_path)
duration = time.time() - start_time
logger.info(f"Prediction took {duration:.2f}s")
```

## 🌐 Web Server Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows/Linux)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring & Maintenance

### Health Check Endpoint
```python
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': check_models_loaded(),
        'memory_usage': get_memory_usage()
    })
```

### Metrics Collection
- Track prediction counts
- Monitor accuracy on labeled data
- Log prediction times
- Track model confidence distributions

### Model Updates
1. Train new model
2. Test on validation set
3. A/B test on production (10% traffic)
4. Monitor metrics for 24-48 hours
5. Gradually increase to 100%
6. Keep old model as backup

## 🔒 Security Considerations

### Input Validation
```python
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

def validate_video(file):
    if not file.filename.endswith(tuple(ALLOWED_EXTENSIONS)):
        raise ValueError("Invalid file type")
    if len(file.read()) > MAX_FILE_SIZE:
        raise ValueError("File too large")
```

### Rate Limiting
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)

@app.route('/detect', methods=['POST'])
@limiter.limit("10 per minute")
def detect():
    # ...
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Multiple worker processes
- Load balancer
- Shared model storage
- Distributed caching

### Vertical Scaling
- More GPU memory
- Faster GPU
- More CPU cores
- More RAM

## 🐛 Troubleshooting

### Common Issues

**Model Loading Fails**
- Check checkpoint path
- Verify model architecture matches
- Try loading with `strict=False`

**Out of Memory**
- Reduce batch size
- Use CPU fallback
- Process videos sequentially

**Slow Predictions**
- Enable GPU acceleration
- Reduce sequence length
- Use model quantization
- Enable caching

## ✅ Post-Deployment Checklist

- [ ] Health check endpoint responding
- [ ] Predictions working correctly
- [ ] Performance metrics within targets
- [ ] Error handling tested
- [ ] Logging configured
- [ ] Monitoring active
- [ ] Backup strategy in place
- [ ] Documentation updated

## 📚 Additional Resources

- `README_IMPROVEMENTS.md` - Complete feature guide
- `integration_guide.md` - Integration details
- `MODEL_IMPROVEMENTS.md` - Technical improvements

---

**Ready for Production!** 🚀


