from typing import Optional
import torch
import torch.nn as nn

class WatermarkFilter(nn.Module):
    """워터마크 바운딩 박스 필터링 모듈"""
    
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, box_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            box_features: 바운딩 박스 특징 (B, num_boxes, feature_dim)
        Returns:
            scores: 워터마크가 아닐 확률 (B, num_boxes)
        """
        return self.filter(box_features).squeeze(-1) 