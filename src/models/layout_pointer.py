import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LayoutPointer(nn.Module):
    """
    Layout Pointer 모듈
    box features와 tag features 간의 attention을 계산하여 
    각 box가 어떤 tag를 가리켜야 하는지, 그리고 빈 셀을 식별합니다.
    """
    def __init__(self, feature_dim: int, temperature: float = 0.1):
        super().__init__()
        # Box와 Tag를 위한 projection layers
        self.box_proj = nn.Linear(feature_dim, feature_dim)
        self.tag_proj = nn.Linear(feature_dim, feature_dim)
        
        # Empty cell을 위한 special embedding (b̄0)
        self.empty_box_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.temperature = temperature
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        # projection layers
        nn.init.xavier_uniform_(self.box_proj.weight)
        nn.init.zeros_(self.box_proj.bias)
        nn.init.xavier_uniform_(self.tag_proj.weight)
        nn.init.zeros_(self.tag_proj.bias)
        
        # empty embedding
        nn.init.normal_(self.empty_box_embedding, mean=0.0, std=0.02)
    
    def forward(self, box_features: torch.Tensor, tag_features: torch.Tensor, 
                data_tag_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            box_features: (B, 994, feature_dim) - bbox embeddings
            tag_features: (B, 30, feature_dim) - tag embeddings
            data_tag_mask: (B, 1024) - C 태그 위치의 mask (앞 994는 layout prompt)
        Returns:
            pointer_logits: (B, 994, 30) - raw logits
            empty_pointer_logits: (B, 30) - raw logits
        """
        B = box_features.size(0)
        
        # 1. Project features
        box_proj = F.normalize(self.box_proj(box_features), dim=-1)  # (B, 994, D)
        tag_proj = F.normalize(self.tag_proj(tag_features), dim=-1)  # (B, 30, D)
        
        # 2. Compute pointer logits for regular boxes (raw scores)
        pointer_logits = torch.matmul(box_proj, tag_proj.transpose(-2, -1))  # (B, 994, 30)
        
        # 3. Compute empty pointer logits (raw scores)
        empty_box_proj = F.normalize(self.box_proj(self.empty_box_embedding), dim=-1)  # (1, 1, D)
        empty_pointer_logits = torch.matmul(
            empty_box_proj.expand(B, -1, -1),  # (B, 1, D)
            tag_proj.transpose(-2, -1)  # (B, D, 30)
        ).squeeze(1)  # (B, 30)
        
        return pointer_logits, empty_pointer_logits