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
        
    def forward(self, decoder_hidden_states: torch.Tensor, num_boxes: int, data_tag_mask: torch.Tensor, empty_mask: torch.Tensor):
        """Layout pointer forward pass
        
        Args:
            decoder_hidden_states: (B, L, D) - decoder의 last hidden states
            num_boxes: int - 실제 bbox 수
            data_tag_mask: (B, L) - 'C' 토큰의 위치
            empty_mask: (B, L) - empty cell의 위치
        """
        # 1. Split features
        box_features = decoder_hidden_states[:, :num_boxes]  # (B, N, D)
        tag_features = decoder_hidden_states[:, num_boxes:]  # (B, T, D)
        
        # 2. Project features (논문 Equation 1)
        box_proj = self.box_proj(box_features)  # (B, N, D)
        tag_proj = self.tag_proj(tag_features)  # (B, T, D)
        
        # Scale 조정을 위해 normalize
        box_proj = F.normalize(box_proj, dim=-1)
        tag_proj = F.normalize(tag_proj, dim=-1)
        
        # 3. Empty cell handling (논문 Equation 3)
        empty_proj = self.empty_proj(self.empty_embedding)  # (1, 1, D)
        empty_logits = torch.matmul(tag_proj, empty_proj.transpose(-1, -2))  # (B, T, 1)
        empty_logits = empty_logits.squeeze(-1)  # (B, T)
        empty_logits = empty_logits / self.temperature
        
        # -inf 대신 매우 작은 값으로 마스킹
        empty_logits = empty_logits.masked_fill(~data_tag_mask, -1e4)
        
        # 4. Calculate pointer logits (논문 Equation 2)
        pointer_logits = torch.matmul(box_proj, tag_proj.transpose(-1, -2))  # (B, N, T)
        pointer_logits = pointer_logits / self.temperature
        
        # 5. Mask invalid positions
        pointer_logits = pointer_logits.masked_fill(~data_tag_mask.unsqueeze(1), float('-inf'))
        
        return pointer_logits, empty_logits