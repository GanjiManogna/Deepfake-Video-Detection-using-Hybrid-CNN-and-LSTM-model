#!/usr/bin/env python3
"""
Batch test multiple videos across all models
Compare CNN-only, LSTM-only, Hybrid, and Ensemble predictions
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def batch_test_videos(video_dir="testing_data", output_file="batch_test_results.json"):
    """Test all videos in directory with all models"""
    video_dir = Path(video_dir)
    output_file = Path(output_file)
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return
    
    # Find all video files
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print(f"⚠️  No video files found in {video_dir}")
        return
    
    print(f"🔍 Found {len(video_files)} videos to test")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_videos": len(video_files),
        "videos": []
    }
    
    # Import predictor
    try:
        from web.simple_tri_predictor import predict_video_simple_three
    except ImportError:
        print("❌ Could not import predictor")
        return
    
    # Test each video
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Testing: {video_path.name}")
        
        try:
            result = predict_video_simple_three(str(video_path))
            
            video_result = {
                "filename": video_path.name,
                "path": str(video_path),
                "predictions": {
                    "cnn_only": result.get("cnn_only", {}),
                    "lstm_only": result.get("lstm_only", {}),
                    "hybrid": result.get("hybrid", {}),
                    "overall": result.get("overall", {})
                },
                "processing_times": result.get("processing_times", {}),
                "errors": result.get("errors", [])
            }
            
            results["videos"].append(video_result)
            
            # Print summary
            overall = result.get("overall", {})
            print(f"  ✅ Overall: {overall.get('prediction', 'UNKNOWN')} "
                  f"({overall.get('confidence', 0):.2%})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["videos"].append({
                "filename": video_path.name,
                "path": str(video_path),
                "error": str(e)
            })
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("📊 Statistics")
    print("=" * 60)
    
    predictions_count = defaultdict(int)
    for video in results["videos"]:
        if "predictions" in video:
            overall_pred = video["predictions"].get("overall", {}).get("prediction")
            if overall_pred:
                predictions_count[overall_pred] += 1
    
    print("\nOverall Predictions:")
    for pred, count in predictions_count.items():
        pct = (count / len(results["videos"])) * 100
        print(f"  {pred}: {count} ({pct:.1f}%)")
    
    # Model agreement analysis
    print("\nModel Agreement:")
    all_agree = 0
    two_agree = 0
    all_disagree = 0
    
    for video in results["videos"]:
        if "predictions" in video:
            preds = video["predictions"]
            cnn_pred = preds.get("cnn_only", {}).get("prediction")
            lstm_pred = preds.get("lstm_only", {}).get("prediction")
            hybrid_pred = preds.get("hybrid", {}).get("prediction")
            
            if cnn_pred and lstm_pred and hybrid_pred:
                if cnn_pred == lstm_pred == hybrid_pred:
                    all_agree += 1
                elif (cnn_pred == lstm_pred) or (cnn_pred == hybrid_pred) or (lstm_pred == hybrid_pred):
                    two_agree += 1
                else:
                    all_disagree += 1
    
    total_with_all = all_agree + two_agree + all_disagree
    if total_with_all > 0:
        print(f"  All agree: {all_agree} ({all_agree/total_with_all*100:.1f}%)")
        print(f"  Two agree: {two_agree} ({two_agree/total_with_all*100:.1f}%)")
        print(f"  All disagree: {all_disagree} ({all_disagree/total_with_all*100:.1f}%)")
    
    # Save results
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Total videos tested: {len(results['videos'])}")
    
    return results

if __name__ == "__main__":
    import sys
    
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "testing_data"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "testing_results/batch_test_results.json"
    
    batch_test_videos(video_dir, output_file)


