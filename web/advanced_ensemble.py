#!/usr/bin/env python3
"""
Advanced Ensemble Fusion for combining CNN, LSTM, and Hybrid predictions
Implements weighted voting, confidence-based fusion, and learnable weights
"""

import numpy as np
import torch
import torch.nn as nn


class AdvancedEnsemble:
    """
    Advanced ensemble fusion with multiple strategies
    """
    
    def __init__(self, strategy='weighted_confidence'):
        """
        Args:
            strategy: 'majority' (simple vote), 'weighted_confidence' (default),
                     'probability_avg' (logit space), 'learned_weights'
        """
        self.strategy = strategy
        self.learned_weights = None
    
    def fuse_majority_vote(self, predictions, confidences=None):
        """Simple majority voting"""
        fake_count = sum(1 for p in predictions if p == 'FAKE' or p == 1)
        real_count = sum(1 for p in predictions if p == 'REAL' or p == 0)
        
        if fake_count > real_count:
            final_pred = 'FAKE'
            final_conf = fake_count / len(predictions)
        else:
            final_pred = 'REAL'
            final_conf = real_count / len(predictions)
        
        return final_pred, final_conf
    
    def fuse_weighted_confidence(self, predictions, confidences):
        """Weight predictions by their confidence scores"""
        fake_vote = 0.0
        real_vote = 0.0
        
        for pred, conf in zip(predictions, confidences):
            pred_val = 1 if (pred == 'FAKE' or pred == 1) else 0
            weighted_conf = conf
            
            if pred_val == 1:
                fake_vote += weighted_conf
            else:
                real_vote += weighted_conf
        
        total_vote = fake_vote + real_vote
        
        if fake_vote > real_vote:
            final_pred = 'FAKE'
            final_conf = fake_vote / total_vote if total_vote > 0 else 0.5
        else:
            final_pred = 'REAL'
            final_conf = real_vote / total_vote if total_vote > 0 else 0.5
        
        return final_pred, final_conf
    
    def fuse_probability_avg(self, probabilities):
        """Average probabilities in logit space (better than simple average)"""
        # Convert probabilities to logits
        logits = [np.log(p / (1 - p + 1e-8)) for p in probabilities]
        
        # Average logits
        avg_logit = np.mean(logits)
        
        # Convert back to probability
        avg_prob = 1 / (1 + np.exp(-avg_logit))
        
        final_pred = 'FAKE' if avg_prob > 0.5 else 'REAL'
        final_conf = avg_prob if final_pred == 'FAKE' else 1 - avg_prob
        
        return final_pred, final_conf
    
    def fuse_learned_weights(self, predictions, confidences, weights):
        """Use pre-learned model weights"""
        fake_vote = 0.0
        real_vote = 0.0
        
        for pred, conf, weight in zip(predictions, confidences, weights):
            pred_val = 1 if (pred == 'FAKE' or pred == 1) else 0
            weighted_conf = conf * weight
            
            if pred_val == 1:
                fake_vote += weighted_conf
            else:
                real_vote += weighted_conf
        
        total_vote = fake_vote + real_vote
        
        if fake_vote > real_vote:
            final_pred = 'FAKE'
            final_conf = fake_vote / total_vote if total_vote > 0 else 0.5
        else:
            final_pred = 'REAL'
            final_conf = real_vote / total_vote if total_vote > 0 else 0.5
        
        return final_pred, final_conf
    
    def fuse(self, results_dict):
        """
        Fuse results from CNN, LSTM, and Hybrid models
        
        Args:
            results_dict: {
                'cnn_only': {'prediction': 'FAKE', 'confidence': 0.85, 'probability': 0.85},
                'lstm_only': {'prediction': 'REAL', 'confidence': 0.72, 'probability': 0.28},
                'hybrid': {'prediction': 'FAKE', 'confidence': 0.90, 'probability': 0.90}
            }
        
        Returns:
            {'prediction': 'FAKE', 'confidence': 0.89, 'method': 'weighted_confidence'}
        """
        predictions = []
        confidences = []
        probabilities = []
        
        for model_key in ['cnn_only', 'lstm_only', 'hybrid']:
            if model_key in results_dict and results_dict[model_key]:
                model_result = results_dict[model_key]
                predictions.append(model_result.get('prediction', 'UNKNOWN'))
                confidences.append(model_result.get('confidence', 0.5))
                probabilities.append(model_result.get('probability', 0.5))
        
        if not predictions:
            return {'prediction': 'UNKNOWN', 'confidence': 0.0, 'method': 'none'}
        
        # Apply fusion strategy
        if self.strategy == 'majority':
            final_pred, final_conf = self.fuse_majority_vote(predictions)
        elif self.strategy == 'weighted_confidence':
            final_pred, final_conf = self.fuse_weighted_confidence(predictions, confidences)
        elif self.strategy == 'probability_avg':
            final_pred, final_conf = self.fuse_probability_avg(probabilities)
        elif self.strategy == 'learned_weights':
            # Default weights: Hybrid=0.5, CNN=0.3, LSTM=0.2 (can be learned)
            default_weights = [0.3, 0.2, 0.5]  # CNN, LSTM, Hybrid
            weights = default_weights[:len(predictions)]
            final_pred, final_conf = self.fuse_learned_weights(predictions, confidences, weights)
        else:
            # Fallback to weighted confidence
            final_pred, final_conf = self.fuse_weighted_confidence(predictions, confidences)
        
        return {
            'prediction': final_pred,
            'confidence': final_conf,
            'method': self.strategy,
            'individual_predictions': {
                'cnn': predictions[0] if len(predictions) > 0 else None,
                'lstm': predictions[1] if len(predictions) > 1 else None,
                'hybrid': predictions[2] if len(predictions) > 2 else None
            },
            'individual_confidences': {
                'cnn': confidences[0] if len(confidences) > 0 else None,
                'lstm': confidences[1] if len(confidences) > 1 else None,
                'hybrid': confidences[2] if len(confidences) > 2 else None
            }
        }


# Default ensemble instance
_default_ensemble = AdvancedEnsemble(strategy='weighted_confidence')


def fuse_three_model_predictions(results_dict, strategy='weighted_confidence'):
    """
    Convenience function to fuse three model predictions
    
    Usage:
        result = fuse_three_model_predictions({
            'cnn_only': cnn_result,
            'lstm_only': lstm_result,
            'hybrid': hybrid_result
        })
    """
    ensemble = AdvancedEnsemble(strategy=strategy)
    return ensemble.fuse(results_dict)


