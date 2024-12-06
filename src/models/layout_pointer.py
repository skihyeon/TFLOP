import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LayoutPointer(nn.Module):
    def __init__(self, feature_dim, temperature):
        super().__init__()
        # Empty cell을 위한 특별한 임베딩
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.empty_proj = nn.Linear(feature_dim, feature_dim)
        
        # Box와 tag projection
        self.box_proj = nn.Linear(feature_dim, feature_dim)
        self.tag_proj = nn.Linear(feature_dim, feature_dim)
        
        self.temperature = temperature
        
    def forward(self, decoder_hidden_states, layout_prompt_length):
        B = decoder_hidden_states.size(0)
        
        # Box와 tag features 분리 및 projection
        box_features = decoder_hidden_states[:, :layout_prompt_length]  # (B, :688, 1024)
        tag_features = decoder_hidden_states[:, layout_prompt_length:]  # (B, 688:, 1024)
        
        box_features = self.box_proj(box_features)  # (B, :688, 1024)
        tag_features = self.tag_proj(tag_features)  # (B, 688:, 1024)
        
        # Empty embedding을 batch size만큼 확장
        empty_embedding = self.empty_embedding.expand(B, -1, -1)  # (B, 1, 1024)
        empty_features = self.empty_proj(empty_embedding)  # (B, 1, 1024)
        
        # Normalize all features
        box_features = F.normalize(box_features, dim=-1)
        tag_features = F.normalize(tag_features, dim=-1)
        empty_features = F.normalize(empty_features, dim=-1)
        
        return box_features, tag_features, empty_features
