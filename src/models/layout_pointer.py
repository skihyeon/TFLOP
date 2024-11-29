import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any
import math

class LayoutPointer(nn.Module):
    """
    Layout Pointer Module
    
    논문 Section 3.5 Layout Pointer 참조:
    - decoder의 last hidden states를 box features와 tag features로 분리
    - linear projection을 통해 box-tag 관계 예측
    - empty cell 처리를 위한 special embedding 사용
    """
    def __init__(
        self, 
        feature_dim: int, 
        temperature: float = 0.1,
        tokenizer: Optional[Any] = None
    ) -> None:
        super().__init__()
        
        # Box and tag projections (논문 Equation 1)
        self.box_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1)
        )
        
        self.tag_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Empty cell handling (논문 Equation 3)
        self.empty_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.empty_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.temperature = temperature
        self.tokenizer = tokenizer
        
        # Initialize parameters
        nn.init.normal_(self.empty_embedding, std=0.02)
        
    def forward(self, decoder_hidden_states: torch.Tensor, num_boxes: int, data_tag_mask: torch.Tensor):
        """
        Args:
            decoder_hidden_states: (B, L, D)
            num_boxes: int
            data_tag_mask: (B, L)
        """
        # 1. Split features
        box_features = decoder_hidden_states[:, :num_boxes]  # (B, N, D)
        tag_features = decoder_hidden_states[:, num_boxes:]  # (B, T, D)
        
        # 2. Project features
        box_proj = self.box_proj(box_features)  # (B, N, D)
        tag_proj = self.tag_proj(tag_features)  # (B, T, D)
        
        # 3. Calculate pointer logits
        pointer_logits = torch.matmul(box_proj, tag_proj.transpose(-1, -2))  # (B, N, T)
        pointer_logits = pointer_logits / self.temperature
        
        # 5. Empty cell handling
        empty_embedding_proj = self.empty_proj(self.empty_embedding)  # (1, 1, D)
        empty_logits = torch.matmul(tag_proj, empty_embedding_proj.transpose(-1, -2)).squeeze(-1)  # (B, T)
        empty_logits = empty_logits / self.temperature
        
        return pointer_logits, empty_logits