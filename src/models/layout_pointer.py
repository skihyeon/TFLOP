import torch
import torch.nn as nn
import torch.nn.functional as F

class LayoutPointer(nn.Module):
    def __init__(self, feature_dim: int, temperature: float):
        super().__init__()
        # 1. Projections for box and tag features (Equation 1)
        self.box_proj = nn.Linear(feature_dim, feature_dim)  # projb
        self.tag_proj = nn.Linear(feature_dim, feature_dim)  # projt
        
        # 2. Special embedding for empty cells (Equation 3)
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.empty_proj = nn.Linear(feature_dim, feature_dim)  # projb for empty
        
        self.temperature = temperature
    
    def forward(self, box_features: torch.Tensor, tag_features: torch.Tensor):
        B = box_features.size(0)
        
        # 1. Project features (Equation 1)
        box_proj = self.box_proj(box_features)  # b̄j = projb(bj)
        tag_proj = self.tag_proj(tag_features)  # t̄k = projt(tk)
        
        # 2. Normalize projected features
        box_proj = F.normalize(box_proj, dim=-1)
        tag_proj = F.normalize(tag_proj, dim=-1)
        
        # 3. Compute pointer logits (for Equation 2)
        pointer_logits = torch.matmul(box_proj, tag_proj.transpose(-2, -1))
        pointer_logits = pointer_logits / self.temperature  # scale by temperature τ
        
        # 4. Handle empty cells (for Equation 3)
        empty_embedding = self.empty_embedding.expand(B, -1, -1)  # b̄0
        empty_proj = self.empty_proj(empty_embedding)
        empty_proj = F.normalize(empty_proj, dim=-1)
        
        empty_pointer_logits = torch.matmul(empty_proj, tag_proj.transpose(-2, -1))
        empty_pointer_logits = empty_pointer_logits / self.temperature
        
        return pointer_logits, empty_pointer_logits