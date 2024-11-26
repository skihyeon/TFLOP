from typing import Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops

class LayoutEncoder(nn.Module):
    """
    Layout Encoder (논문 3.3)
    - MLP modules for embedding text region bounding boxes
    - 2x2 ROIAlign for visual features
    - Aggregation to form layout embeddings {lj|lj ∈ Rd,1 ≤ j ≤ B}
    """
    def __init__(
        self, 
        feature_dim: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Bounding box MLP encoder
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ROI Align module (논문: 2x2)
        self.roi_align = ops.RoIAlign(
            output_size=(2, 2),  # 논문에 명시된 2x2 
            spatial_scale=1.0,
            sampling_ratio=2
        )
        
        # ROI feature MLP encoder
        roi_feat_size = 2 * 2 * feature_dim  # 2x2 ROI align output
        self.roi_mlp = nn.Sequential(
            nn.Linear(roi_feat_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final aggregation MLP
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        bboxes: torch.Tensor,  # (B, N, 4) - N text region bounding boxes
        visual_features: torch.Tensor  # (B, P, D) - P patches of visual features
    ) -> torch.Tensor:
        """
        Args:
            bboxes: text region bounding boxes
            visual_features: image encoder의 visual features {zi}
        Returns:
            layout_embeddings: {lj|lj ∈ Rd,1 ≤ j ≤ B} (B, N, D)
        """
        B, N = bboxes.size()[:2]
        H = W = int(visual_features.size(1) ** 0.5)  # Assume square feature map
        D = visual_features.size(-1)
        
        # 1. Embed bounding boxes
        bbox_features = self.bbox_mlp(bboxes)  # (B, N, D)
        
        # 2. Apply ROI Align to get visual features
        visual_features = visual_features.transpose(1, 2).view(B, D, H, W)
        batch_idx = torch.arange(B, device=bboxes.device).view(-1, 1).repeat(1, N).view(-1)
        rois = torch.cat([batch_idx.unsqueeze(1), bboxes.view(-1, 4)], dim=1)
        roi_features = self.roi_align(visual_features, rois)  # (B*N, D, 2, 2)
        roi_features = roi_features.view(B*N, -1)  # Flatten ROI features
        roi_features = self.roi_mlp(roi_features).view(B, N, -1)  # (B, N, D)
        
        # 3. Aggregate embeddings
        layout_embeddings = self.aggregation_mlp(
            torch.cat([bbox_features, roi_features], dim=-1)
        )  # (B, N, D)
        
        return layout_embeddings