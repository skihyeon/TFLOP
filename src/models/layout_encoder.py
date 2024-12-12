import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class LayoutEncoder(nn.Module):
    def __init__(self, feature_dim: int, input_size: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Bounding box MLP encoder
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, feature_dim),  # normalized coordinates (x1, y1, x2, y2)
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 2x2 ROI Align (논문 3.3 참조)
        self.roi_align = ops.RoIAlign(
            output_size=(2, 2),        # 논문에서 명시한 2x2
            spatial_scale=1.0/32.0,    # Swin의 최종 feature map scale
            sampling_ratio=2,
            aligned=True
        )
        
        # 3. ROI feature projection
        self.roi_proj = nn.Sequential(
            nn.Flatten(),              # 2x2xD -> 4D
            nn.Linear(4 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 4. Final aggregation layer
        self.aggregate = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),  # bbox_feat + roi_feat
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def prepare_rois(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Convert bboxes to ROI format (batch_idx, x1, y1, x2, y2)"""
        B, N = bboxes.size()[:2]
        batch_idx = torch.arange(B, device=bboxes.device).view(-1, 1).repeat(1, N).view(-1, 1)
        return torch.cat([batch_idx, bboxes.view(-1, 4)], dim=1)
    
    def forward(self, bboxes: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: (B, N, 4) normalized bbox coordinates
            visual_features: (B, H*W, D) visual features from image encoder
        Returns:
            layout_embedding: (B, N, D) layout embeddings
        """
        B, N = bboxes.size()[:2]
        
        # 1. Encode bounding boxes
        bbox_feats = self.bbox_encoder(bboxes)  # (B, N, D)
        
        # 2. Prepare visual features for ROI Align
        H = W = int(visual_features.size(1) ** 0.5)
        visual_feats = visual_features.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 3. Extract ROI features
        rois = self.prepare_rois(bboxes)  # (B*N, 5)
        roi_feats = self.roi_align(visual_feats, rois)  # (B*N, D, 2, 2)
        roi_feats = self.roi_proj(roi_feats).view(B, N, -1)  # (B, N, D)
        
        # 4. Aggregate bbox and ROI features
        layout_embedding = self.aggregate(torch.cat([bbox_feats, roi_feats], dim=-1))  # (B, N, D)
        
        return layout_embedding