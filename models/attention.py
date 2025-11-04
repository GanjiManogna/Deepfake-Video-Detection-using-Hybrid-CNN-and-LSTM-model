"""
Attention mechanisms for improved deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on important frames"""
    
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, features):
        # features: (batch_size, seq_len, hidden_size)
        attention_weights = self.attention(features)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over sequence length
        attended_features = torch.bmm(attention_weights.transpose(1, 2), features)  # (batch_size, 1, hidden_size)
        return attended_features.squeeze(1), attention_weights

class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important regions in frames"""
    
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        )
        
    def forward(self, features):
        # features: (batch_size, channels, height, width)
        attention_map = self.conv(features)  # (batch_size, 1, height, width)
        attention_map = F.softmax(attention_map.view(attention_map.size(0), -1), dim=1)
        attention_map = attention_map.view(attention_map.size(0), 1, features.size(2), features.size(3))
        attended_features = features * attention_map
        return attended_features, attention_map

class MultiModalAttention(nn.Module):
    """Combines different feature modalities with attention"""
    
    def __init__(self, spatial_dim, temporal_dim, num_modalities=2):
        super(MultiModalAttention, self).__init__()
        self.spatial_projection = nn.Linear(spatial_dim, temporal_dim)
        self.modality_attention = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, num_modalities)
        )
        
    def forward(self, spatial_features, temporal_features):
        # Project spatial features to match temporal dimension
        spatial_features = self.spatial_projection(spatial_features)
        combined = torch.cat([spatial_features, temporal_features], dim=1)
        weights = F.softmax(self.modality_attention(combined), dim=1)
        attended = weights[:, 0].unsqueeze(1) * spatial_features + \
                  weights[:, 1].unsqueeze(1) * temporal_features
        return attended, weights