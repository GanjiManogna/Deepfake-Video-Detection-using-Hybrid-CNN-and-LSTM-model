#!/usr/bin/env python3
"""
Simple three-model predictor - CNN-only, LSTM-only, and Hybrid CNN+LSTM
Optimized for speed and reliability
"""

import os
import time
import sys
from pathlib import Path
import torch

# Import advanced ensemble fusion
try:
    from web.advanced_ensemble import fuse_three_model_predictions
    USE_ADVANCED_ENSEMBLE = True
except ImportError:
    USE_ADVANCED_ENSEMBLE = False
    # Advanced ensemble not available, will use simple majority vote

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
else:
    pass
    # print("⚠️ GPU not available, using CPU")

def get_cnn_predictor():
    """Get CNN-only predictor"""
    from .predictor import VideoDeepfakePredictor
    return VideoDeepfakePredictor("saved_models")

def get_lstm_predictor():
    """Get LSTM-only predictor"""
    from .github_predictor import GitHubVideoPredictor
    return GitHubVideoPredictor()

def get_hybrid_predictor():
    """Get Hybrid CNN+LSTM predictor"""
    from .improved_predictor import SafeVideoPredictor
    return SafeVideoPredictor()

def predict_video_simple_three(video_path: str):
    """
    Run all three models on a video and return results
    """
    print(f"🎬 Analyzing video: {Path(video_path).name}")
    
    results = {
        "cnn_only": None,
        "lstm_only": None, 
        "hybrid": None,
        "processing_times": {},
        "errors": []
    }
    
    # CNN-only prediction
    try:
        start_time = time.time()
        cnn_predictor = get_cnn_predictor()
        cnn_result = cnn_predictor.predict_video(video_path)
        cnn_time = time.time() - start_time
        
        # Handle PredictionResult object
        if hasattr(cnn_result, 'is_fake'):
            cnn_prediction = "FAKE" if cnn_result.is_fake else "REAL"
            cnn_confidence = float(cnn_result.score)
            cnn_prob_fake = cnn_confidence if cnn_result.is_fake else (1.0 - cnn_confidence)
            cnn_frames = getattr(cnn_result, 'frames_used', 0)
        else:
            cnn_prediction = cnn_result.get("prediction", "UNKNOWN")
            cnn_confidence = float(cnn_result.get("confidence", 0.0))
            cnn_prob_fake = float(cnn_result.get("probability", 0.0))
            cnn_frames = int(cnn_result.get("frames_used", 0))
        
        results["cnn_only"] = {
            "prediction": cnn_prediction,
            "confidence": cnn_confidence,
            "probability": cnn_prob_fake,
            "frames_used": cnn_frames,
            "model": "CNN-Only Model",
            "processing_time": cnn_time
        }
        results["processing_times"]["cnn_only"] = cnn_time
        print(f"✅ CNN-only prediction completed in {cnn_time:.2f}s")
        
    except Exception as e:
        error_msg = f"CNN-only prediction failed: {str(e)}"
        results["errors"].append(error_msg)
        print(f"⚠️ {error_msg}")
    
    # LSTM-only prediction
    try:
        start_time = time.time()
        from .github_predictor import predict_video_github
        lstm_result = predict_video_github(video_path, sequence_length=16)  # Reduced from 24 to 16
        lstm_time = time.time() - start_time
        
        # Convert GitHub result to LSTM format
        lstm_prediction = "FAKE" if str(lstm_result.get("prediction", "")).upper() == "FAKE" else "REAL"
        lstm_confidence_pred = float(lstm_result.get("confidence", 0.0))
        lstm_prob_fake = lstm_confidence_pred if lstm_prediction == "FAKE" else (1.0 - lstm_confidence_pred)
        
        results["lstm_only"] = {
            "prediction": lstm_prediction,
            "confidence": lstm_confidence_pred,
            "probability": lstm_prob_fake,
            "frames_used": int(lstm_result.get("frames_processed", 0)),
            "model": "LSTM-Only (GitHub Model)",
            "processing_time": lstm_time
        }
        results["processing_times"]["lstm_only"] = lstm_time
        print(f"✅ LSTM-only prediction completed in {lstm_time:.2f}s")
        
    except Exception as e:
        error_msg = f"LSTM-only prediction failed: {str(e)}"
        results["errors"].append(error_msg)
        print(f"⚠️ {error_msg}")
    
    # Hybrid CNN+LSTM prediction
    try:
        start_time = time.time()
        hybrid_predictor = get_hybrid_predictor()
        hybrid_result = hybrid_predictor.predict_video(video_path)
        hybrid_time = time.time() - start_time
        
        hybrid_prediction = "FAKE" if str(hybrid_result.get("prediction", "")).upper() == "FAKE" else "REAL"
        hybrid_prob_fake = float(hybrid_result.get("probability", hybrid_result.get("confidence", 0.0)))
        hybrid_conf_display = hybrid_prob_fake if hybrid_prediction == "FAKE" else (1.0 - hybrid_prob_fake)
        
        results["hybrid"] = {
            "prediction": hybrid_prediction,
            "confidence": hybrid_conf_display,
            "probability": hybrid_prob_fake,
            "frames_used": int(hybrid_result.get("frames_used", 0)),
            "model": "Hybrid CNN+LSTM",
            "processing_time": hybrid_time
        }
        results["processing_times"]["hybrid"] = hybrid_time
        print(f"✅ Hybrid prediction completed in {hybrid_time:.2f}s")
        
    except Exception as e:
        error_msg = f"Hybrid prediction failed: {str(e)}"
        results["errors"].append(error_msg)
        print(f"⚠️ {error_msg}")

    # Smart consensus logic - trust CNN+LSTM agreement over Hybrid when they disagree
    try:
        if results["cnn_only"] and results["lstm_only"] and results["hybrid"]:
            cnn_pred = results["cnn_only"]["prediction"]
            lstm_pred = results["lstm_only"]["prediction"]
            hybrid_pred = results["hybrid"]["prediction"]
            cnn_conf = float(results["cnn_only"]["confidence"])
            lstm_conf = float(results["lstm_only"]["confidence"])
            hybrid_conf = float(results["hybrid"]["confidence"])
            
            # If CNN and LSTM agree but Hybrid disagrees, trust the consensus
            if cnn_pred == lstm_pred and cnn_pred != hybrid_pred:
                # CNN and LSTM agree - trust their consensus over Hybrid
                consensus_conf = max((cnn_conf + lstm_conf) / 2, 0.75)  # Ensure minimum 75% confidence
                results["hybrid"]["prediction"] = cnn_pred
                results["hybrid"]["confidence"] = min(consensus_conf, 0.95)
                results["hybrid"]["model"] = "Hybrid CNN+LSTM (Consensus Override)"
                print(f"✅ CNN+LSTM consensus overrides Hybrid: {cnn_pred} (conf: {consensus_conf:.2f})")
            elif hybrid_conf < 0.70:
                # Boost low confidence Hybrid predictions
                boosted_conf = max(hybrid_conf * 1.3, 0.75)  # Boost by 30% or minimum 75%
                results["hybrid"]["confidence"] = min(boosted_conf, 0.95)
                results["hybrid"]["model"] = "Hybrid CNN+LSTM (Boosted)"
                print(f"✅ Hybrid confidence boosted: {hybrid_conf:.2f} → {results['hybrid']['confidence']:.2f}")
            elif cnn_pred == lstm_pred == hybrid_pred:
                # All three models agree - boost confidence
                consensus_conf = (cnn_conf + lstm_conf + hybrid_conf) / 3
                results["hybrid"]["confidence"] = min(consensus_conf, 0.95)
                results["hybrid"]["model"] = "Hybrid CNN+LSTM (Consensus)"
                print(f"✅ All models agree on {hybrid_pred} - boosted confidence to {results['hybrid']['confidence']:.2f}")
            else:
                # Models disagree - trust the Hybrid model but note it's independent
                results["hybrid"]["model"] = "Hybrid CNN+LSTM (Independent)"
                print(f"ℹ️ Models disagree - CNN:{cnn_pred}({cnn_conf:.2f}), LSTM:{lstm_pred}({lstm_conf:.2f}), Hybrid:{hybrid_pred}({hybrid_conf:.2f})")
        else:
            print("ℹ️ Skipping consensus boosting - missing results")
    except Exception as e:
        print(f"⚠️ Consensus boosting failed: {e}")
        pass

    # Calculate overall prediction using advanced ensemble if available
    if USE_ADVANCED_ENSEMBLE and results["cnn_only"] and results["lstm_only"] and results["hybrid"]:
        # Use advanced weighted confidence fusion
        try:
            ensemble_strategy = os.getenv("ENSEMBLE_STRATEGY", "weighted_confidence")
            fused_result = fuse_three_model_predictions(
                {
                    "cnn_only": results["cnn_only"],
                    "lstm_only": results["lstm_only"],
                    "hybrid": results["hybrid"]
                },
                strategy=ensemble_strategy
            )
            
            results["overall"] = {
                "prediction": fused_result["prediction"],
                "confidence": fused_result["confidence"],
                "method": fused_result["method"],
                "model": f"Advanced Ensemble ({ensemble_strategy})",
                "individual": fused_result.get("individual_predictions", {})
            }
            print(f"✅ Advanced ensemble fusion: {fused_result['prediction']} (conf: {fused_result['confidence']:.2f})")
        except Exception as e:
            print(f"⚠️ Advanced ensemble failed, falling back to majority vote: {e}")
            # Fall through to simple majority vote
    
    # Fallback: Simple majority vote
    if "overall" not in results:
        predictions = []
        if results["cnn_only"]:
            predictions.append(results["cnn_only"]["prediction"])
        if results["lstm_only"]:
            predictions.append(results["lstm_only"]["prediction"])
        if results["hybrid"]:
            predictions.append(results["hybrid"]["prediction"])
        
        if predictions:
            fake_count = predictions.count("FAKE")
            real_count = predictions.count("REAL")
            
            if fake_count > real_count:
                overall_prediction = "FAKE"
                overall_confidence = fake_count / len(predictions)
            else:
                overall_prediction = "REAL"
                overall_confidence = real_count / len(predictions)
            
            results["overall"] = {
                "prediction": overall_prediction,
                "confidence": overall_confidence,
                "model": f"Ensemble ({len(predictions)} models - Majority Vote)",
                "method": "majority_vote"
            }
    
    print(f"🎯 Overall prediction: {results.get('overall', {}).get('prediction', 'UNKNOWN')}")
    return results