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
        # FFN으로 확장하여 표현력 강화
        self.box_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.tag_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Empty cell embedding도 동일한 구조로
        self.empty_box_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.empty_box_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.temperature = temperature
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        # projection layers
        nn.init.xavier_uniform_(self.box_proj[0].weight)
        nn.init.zeros_(self.box_proj[0].bias)
        nn.init.xavier_uniform_(self.tag_proj[0].weight)
        nn.init.zeros_(self.tag_proj[0].bias)
        
        # empty embedding
        nn.init.normal_(self.empty_box_embedding, mean=0.0, std=0.02)
    
    def forward(self, box_features: torch.Tensor, tag_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        box_proj = self.box_proj(box_features)  # (B, 994, D)
        tag_proj = self.tag_proj(tag_features)  # (B, 30, D)
        
        # 2. L2 normalize
        box_proj = F.normalize(box_proj, p=2, dim=-1)
        tag_proj = F.normalize(tag_proj, p=2, dim=-1)
        
        # 3. Compute pointer logits (values will be in [-1, 1])
        pointer_logits = torch.matmul(box_proj, tag_proj.transpose(-2, -1))  # (B, 994, 30)
        
        # 4. Compute empty pointer logits (raw scores)
        empty_box_proj = F.normalize(self.empty_box_proj(self.empty_box_embedding), p=2, dim=-1)  # (1, 1, D)
        empty_pointer_logits = torch.matmul(
            empty_box_proj.expand(B, -1, -1),  # (B, 1, D)
            tag_proj.transpose(-2, -1)  # (B, D, 30)
        ).squeeze(1)  # (B, 30)
        
        # # Debug pointer distributions
        # print("\n=== Layout Pointer Debug ===")
        # print(f"Pointer logits range: [{pointer_logits.min():.4f}, {pointer_logits.max():.4f}]")
        # print(f"Pointer logits mean/std: {pointer_logits.mean():.4f}/{pointer_logits.std():.4f}")
        
        # # Empty pointer stats
        # print(f"Empty pointer range: [{empty_pointer_logits.min():.4f}, {empty_pointer_logits.max():.4f}]")
        # print(f"Empty pointer mean/std: {empty_pointer_logits.mean():.4f}/{empty_pointer_logits.std():.4f}")
        
        # # Attention sparsity check (얼마나 sharp한지)
        # with torch.no_grad():
        #     pointer_probs = (pointer_logits / self.temperature).softmax(dim=-1)
        #     entropy = -(pointer_probs * pointer_probs.log()).sum(dim=-1).mean()
        #     print(f"Attention entropy: {entropy:.4f}")
        
        return pointer_logits, empty_pointer_logits