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
        self.empty_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.empty_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.temperature = temperature
        self.tokenizer = tokenizer
        
        # Initialize parameters
        nn.init.normal_(self.empty_token, std=0.02)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,  # (B, L, D)
        num_boxes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Layout Pointer Mechanism (논문 3.5)
        
        Args:
            decoder_hidden_states: 디코더의 마지막 hidden states (B, L, D)
            num_boxes: 배치 내 최대 박스 수
            
        Returns:
            pointer_logits: box-tag 관계 점수 (B, N, T)
            empty_logits: empty cell 점수 (B, T)
            data_tag_mask: 데이터 태그 위치 마스크 (B, T)
        """
        B, L, D = decoder_hidden_states.size()
        T = L - num_boxes  # 실제 태그 시퀀스 길이
        
        # 1. Split features (논문 3.5)
        box_features = decoder_hidden_states[:, :num_boxes]  # (B, N, D)
        tag_features = decoder_hidden_states[:, num_boxes:]  # (B, T, D)
        
        # 2. Project features (논문 Equation 1)
        box_proj = self.box_proj(box_features)  # (B, N, D)
        tag_proj = self.tag_proj(tag_features)  # (B, T, D)
        
        # 3. Calculate pointer logits (논문 Equation 2)
        pointer_logits = torch.matmul(
            box_proj,  # (B, N, D)
            tag_proj.transpose(-1, -2)  # (B, D, T)
        ) / self.temperature  # (B, N, T)
        
        # 4. Empty cell handling (논문 Equation 3)
        empty_token_proj = self.empty_proj(
            self.empty_token.expand(B, -1, -1)
        )  # (B, 1, D)
        
        empty_logits = torch.matmul(
            tag_proj,  # (B, T, D)
            empty_token_proj.transpose(-1, -2)  # (B, D, 1)
        ).squeeze(-1) / self.temperature  # (B, T)
        
        # 5. Create data tag mask
        if self.tokenizer is not None:
            # OTSL 'C' 태그 ID 가져오기 (data cell)
            data_tag_id = self.tokenizer.token2id.get('C', 2)
            
            # tag_features에서 data tag positions 찾기
            tag_probs = F.softmax(tag_proj @ tag_proj.transpose(-2, -1), dim=-1)  # (B, T, T)
            data_tag_mask = torch.zeros(B, T, dtype=torch.bool, device=decoder_hidden_states.device)
            
            # 'C' 토큰 위치 마스킹
            for b in range(B):
                for t in range(T):
                    if tag_probs[b, t].max() > 0.5:  # confidence threshold
                        data_tag_mask[b, t] = True
        else:
            data_tag_mask = torch.ones(B, T, dtype=torch.bool, device=decoder_hidden_states.device)
        
        return pointer_logits, empty_logits, data_tag_mask