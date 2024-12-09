from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math

class LayoutEncoder(nn.Module):
    def __init__(self, feature_dim: int, input_size: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Spatial encoding (bbox -> embedding)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout)
        )
        
        # 2. Multi-scale ROI feature processing
        scales = [1/8, 1/16, 1/32]
        self.roi_scales = nn.ModuleList([
            ops.RoIAlign(
                output_size=(4, 4),
                spatial_scale=scale,
                sampling_ratio=2,
                aligned=True
            ) for scale in scales
        ])
        
        # 3. Scale-wise feature projection
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 1),
                nn.LayerNorm([feature_dim, 4, 4]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(len(scales))
        ])
        
        # 4. ROI feature encoder
        self.roi_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * feature_dim, feature_dim),  # 4x4 = 16
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout)
        )
        
        # 5. Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, bboxes: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: Tensor of shape (B, N, 4) containing normalized bbox coordinates
            visual_features: Tensor of shape (B, H*W, D) from image encoder
        Returns:
            layout_feats: Tensor of shape (B, N, D) containing layout embeddings
        """
        B, N = bboxes.size()[:2]
        
        # 1. Spatial encoding
        spatial_feats = self.spatial_encoder(bboxes)  # (B, N, D)
        
        # 2. Visual feature preprocessing
        H = W = int(math.sqrt(visual_features.size(1)))
        visual_feats = visual_features.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 3. Multi-scale ROI feature extraction
        rois = self.prepare_rois(bboxes)  # (B*N, 5)
        multi_scale_feats = []
        
        for roi_align, proj in zip(self.roi_scales, self.scale_projections):
            # Extract features at each scale
            scale_feat = roi_align(visual_feats, rois)  # (B*N, D, 4, 4)
            scale_feat = proj(scale_feat)
            multi_scale_feats.append(scale_feat)
        
        # 4. Combine multi-scale features with max pooling
        roi_feats = torch.max(torch.stack(multi_scale_feats), dim=0)[0]  # (B*N, D, 4, 4)
        
        # 5. ROI feature encoding
        roi_feats = self.roi_encoder(roi_feats).view(B, N, -1)  # (B, N, D)
        
        # 6. Combine spatial and ROI features
        layout_feats = self.fusion(
            torch.cat([spatial_feats, roi_feats], dim=-1)
        )  # (B, N, D)
        
        return layout_feats
    
    def prepare_rois(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Convert bboxes to ROI format (batch_idx, x1, y1, x2, y2)"""
        B, N = bboxes.size()[:2]
        batch_idx = torch.arange(B, device=bboxes.device).view(-1, 1).repeat(1, N).view(-1, 1)
        return torch.cat([batch_idx, bboxes.view(-1, 4)], dim=1)
    
    def visualize_layout(
        self,
        image: torch.Tensor,
        bboxes: torch.Tensor,
        visual_features: torch.Tensor,
        save_dir: str = "debug/layout_vis"
    ) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from pathlib import Path
        import numpy as np
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            B = image.size(0)
            for b in range(B):
                # 1. 원본 이미지 시각화
                img = image[b].detach().permute(1,2,0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                
                fig, axes = plt.subplots(3, 2, figsize=(20, 30))
                
                # 원본 이미지 + bbox
                axes[0,0].imshow(img)
                for bbox in bboxes[b].detach():
                    x1, y1, x2, y2 = bbox.cpu().numpy()
                    w, h = x2-x1, y2-y1
                    rect = patches.Rectangle((x1*img.shape[1], y1*img.shape[0]), 
                                        w*img.shape[1], h*img.shape[0],
                                        linewidth=1, edgecolor='r', facecolor='none')
                    axes[0,0].add_patch(rect)
                axes[0,0].set_title("Original Image with Bounding Boxes")
                axes[0,0].axis('off')
                
                # 2. Visual Features 히트맵
                H = W = int(visual_features.size(1) ** 0.5)
                feat_map = visual_features[b].detach().view(H, W, -1).norm(dim=-1)
                im = axes[0,1].imshow(feat_map.cpu().numpy(), cmap='viridis')
                axes[0,1].set_title("Visual Features Norm")
                plt.colorbar(im, ax=axes[0,1])
                axes[0,1].axis('off')
                
                # 3. Multi-scale ROI Align 결과 분석
                visual_feats = visual_features[b].detach()
                visual_feats = visual_feats.transpose(0,1).view(1, -1, H, W)
                
                # Pre-ROI Align stats
                pre_roi_norm = visual_feats.norm(dim=1).squeeze()
                pre_roi_stats = {
                    'mean': pre_roi_norm.mean().item(),
                    'std': pre_roi_norm.std().item(),
                    'spatial_std': pre_roi_norm.std(dim=0).mean().item()
                }
                
                # Multi-scale ROI Align 적용
                rois = self.prepare_rois(bboxes[b:b+1])
                multi_scale_feats = []
                
                for scale_idx, (roi_align, proj) in enumerate(zip(self.roi_scales, self.scale_projections)):
                    scale_feat = roi_align(visual_feats, rois)
                    scale_feat = proj(scale_feat)
                    multi_scale_feats.append(scale_feat)
                    
                    # Scale별 ROI feature 시각화
                    scale_norms = scale_feat.norm(dim=1).mean(dim=(1,2))
                    ax = axes[1, scale_idx % 2]
                    bars = ax.bar(range(len(scale_norms)), scale_norms.cpu().numpy())
                    ax.set_title(f"Scale 1/{2**(scale_idx+3)} ROI Features Norm")
                    ax.set_xlabel("Box Index")
                    ax.set_ylabel("Feature Norm")
                
                # 4. Combined ROI Features 분석
                combined_feats = torch.max(torch.stack(multi_scale_feats), dim=0)[0]
                roi_norms = combined_feats.norm(dim=1).mean(dim=(1,2))
                
                mean_norm = roi_norms.mean().item()
                std_norm = roi_norms.std().item()
                
                # Combined features bar plot
                ax = axes[2,0]
                bars = ax.bar(range(len(roi_norms)), roi_norms.cpu().numpy())
                ax.axhline(y=mean_norm, color='r', linestyle='--', 
                        label=f'Mean: {mean_norm:.2f}')
                ax.axhline(y=mean_norm + std_norm, color='g', linestyle=':', 
                        label=f'Mean ± Std')
                ax.axhline(y=mean_norm - std_norm, color='g', linestyle=':')
                
                # Bar color based on deviation
                for idx, bar in enumerate(bars):
                    norm_val = roi_norms[idx].item()
                    if norm_val > mean_norm + 2*std_norm:
                        bar.set_color('red')
                    elif norm_val < mean_norm - 2*std_norm:
                        bar.set_color('blue')
                
                ax.set_title("Combined Multi-scale ROI Features Norm")
                ax.set_xlabel("Box Index")
                ax.set_ylabel("Feature Norm")
                ax.legend()
                
                # 5. Layout Embedding 시각화
                layout_emb = self.forward(bboxes[b:b+1], visual_features[b:b+1])
                layout_emb = layout_emb.squeeze(0).detach().cpu().numpy()
                
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                layout_emb_2d = pca.fit_transform(layout_emb)
                
                ax = axes[2,1]
                scatter = ax.scatter(layout_emb_2d[:,0], layout_emb_2d[:,1], 
                                c=roi_norms.cpu().numpy(), cmap='viridis')
                for i in range(len(layout_emb_2d)):
                    ax.annotate(str(i), (layout_emb_2d[i,0], layout_emb_2d[i,1]))
                plt.colorbar(scatter, ax=ax, label='ROI Feature Norm')
                ax.set_title("Layout Embeddings (PCA)")
                
                plt.tight_layout()
                plt.savefig(save_dir / f"layout_vis_batch_{b}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 통계 정보 저장
                # 통계 정보 저장 부분만 수정
                with open(save_dir / f"roi_analysis_batch_{b}.txt", "w", encoding='utf-8') as f:
                    # 1. Pre-ROI 특징 분석
                    f.write("=== Pre-ROI Feature Analysis ===\n")
                    f.write(f"Shape: {visual_feats.shape}\n")  # feature map 크기 확인
                    for k, v in pre_roi_stats.items():
                        f.write(f"{k}: {v:.6f}\n")
                    
                    # Feature activation 분석
                    active_ratio = (visual_feats.abs() > visual_feats.abs().mean()).float().mean().item()
                    f.write(f"Active channels ratio: {active_ratio:.4f}\n")  # 활성화된 채널 비율
                    f.write(f"Zero ratio: {(visual_feats.abs() < 1e-6).float().mean().item():.4f}\n")  # 0에 가까운 값들의 비율
                    
                    # 2. Multi-scale ROI 분석
                    f.write("\n=== Multi-scale ROI Analysis ===\n")
                    for scale_idx, scale_feat in enumerate(multi_scale_feats):
                        scale = 1/(2**(scale_idx+3))
                        scale_norms = scale_feat.norm(dim=1).mean(dim=(1,2))
                        
                        f.write(f"\nScale 1/{2**(scale_idx+3)} Analysis:\n")
                        f.write(f"Feature shape: {scale_feat.shape}\n")
                        f.write(f"Mean norm: {scale_norms.mean().item():.4f}\n")
                        f.write(f"Std norm: {scale_norms.std().item():.4f}\n")
                        f.write(f"Min norm: {scale_norms.min().item():.4f}\n")
                        f.write(f"Max norm: {scale_norms.max().item():.4f}\n")
                        
                        # 채널별 통계
                        channel_means = scale_feat.mean(dim=(2,3))  # (B*N, D)
                        channel_stds = scale_feat.std(dim=(2,3))    # (B*N, D)
                        f.write(f"Channel mean range: [{channel_means.min().item():.4f}, {channel_means.max().item():.4f}]\n")
                        f.write(f"Channel std range: [{channel_stds.min().item():.4f}, {channel_stds.max().item():.4f}]\n")
                        
                        # Spatial 분포
                        spatial_var = scale_feat.var(dim=(2,3)).mean().item()  # spatial variation
                        f.write(f"Avg spatial variance: {spatial_var:.4f}\n")
                    
                    # 3. Combined Features 분석
                    f.write("\n=== Combined Multi-scale Features Analysis ===\n")
                    f.write(f"Shape after combination: {combined_feats.shape}\n")
                    f.write(f"Mean norm: {mean_norm:.4f}\n")
                    f.write(f"Std norm: {std_norm:.4f}\n")
                    f.write(f"Min norm: {roi_norms.min().item():.4f}\n")
                    f.write(f"Max norm: {roi_norms.max().item():.4f}\n")
                    
                    # Quartiles for outlier detection
                    percentiles = torch.tensor([0.25, 0.5, 0.75], device=roi_norms.device)
                    quantiles = torch.quantile(roi_norms, percentiles)
                    f.write(f"Quartiles: {quantiles.tolist()}\n")
                    
                    # Outlier boxes
                    outlier_threshold = 2 * std_norm
                    outlier_boxes = torch.where(torch.abs(roi_norms - mean_norm) > outlier_threshold)[0]
                    if len(outlier_boxes) > 0:
                        f.write("\nOutlier boxes (>2std):\n")
                        for box_idx in outlier_boxes:
                            f.write(f"Box {box_idx}: norm={roi_norms[box_idx].item():.4f}\n")
                    
                    # 4. Final Layout Embedding 분석
                    f.write("\n=== Layout Embedding Analysis ===\n")
                    f.write(f"Embedding shape: {layout_emb.shape}\n")
                    f.write(f"Embedding mean: {layout_emb.mean():.4f}\n")
                    f.write(f"Embedding std: {layout_emb.std():.4f}\n")
                    
                    # PCA 분석
                    f.write("\nPCA Analysis:\n")
                    f.write(f"Explained variance ratio: {pca.explained_variance_ratio_.tolist()}\n")
                    f.write(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}\n")
                    
                    # 5. 잠재적 문제점 체크
                    f.write("\n=== Potential Issues ===\n")
                    
                    # Feature collapse 체크
                    if std_norm < 1e-4:
                        f.write("WARNING: Possible feature collapse detected (very low std)\n")
                    
                    # Gradient vanishing/exploding 체크
                    if mean_norm > 100 or mean_norm < 0.01:
                        f.write("WARNING: Unusual feature magnitudes detected\n")
                    
                    # 불균형한 scale contribution 체크
                    scale_contributions = [sf.norm().item() for sf in multi_scale_feats]
                    max_scale_ratio = max(scale_contributions) / (min(scale_contributions) + 1e-6)
                    if max_scale_ratio > 10:
                        f.write(f"WARNING: Unbalanced scale contributions (ratio: {max_scale_ratio:.2f})\n")