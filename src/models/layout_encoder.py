import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class LayoutEncoder(nn.Module):
    def __init__(self, feature_dim: int, input_size: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.spatial_scale = 1.0 / 32.0
        
        # 1. Bbox MLP encoder
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. ROI Align + MLP
        self.roi_align = ops.RoIAlign(
            output_size=(2, 2),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True
        )
        
        # ROI features MLP
        self.roi_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Final MLP for aggregation
        self.aggregate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, bboxes: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        B, N = bboxes.size()[:2]
        
        # 1. Encode bounding boxes through MLP
        bbox_feats = self.bbox_encoder(bboxes)  # (B, N, D)
        
        # 2. Extract and encode ROI features
        # visual_features: (B, D, H, W) where D is feature_dim
        visual_feats = visual_features.view(B, self.feature_dim, self.input_size//32, self.input_size//32)
        
        rois = self.prepare_rois(bboxes)
        roi_feats = self.roi_align(visual_feats, rois)  # (B*N, D, 2, 2)
        roi_feats = roi_feats.flatten(1)  # (B*N, D*4)
        roi_feats = self.roi_mlp(roi_feats)  # (B*N, D)
        roi_feats = roi_feats.view(B, N, -1)  # (B, N, D)
        
        # 3. Aggregate bbox and ROI features
        combined_feats = torch.cat([bbox_feats, roi_feats], dim=-1)  # (B, N, 2D)
        layout_embedding = self.aggregate(combined_feats)  # (B, N, D)
        
        return layout_embedding
    
    def prepare_rois(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Convert bboxes to ROI format (batch_idx, x1, y1, x2, y2)"""
        B, N = bboxes.size()[:2]
        batch_idx = torch.arange(B, device=bboxes.device).view(-1, 1).repeat(1, N).view(-1, 1)
        return torch.cat([batch_idx, bboxes.view(-1, 4)], dim=1)