import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from typing import Optional, Dict
from pathlib import Path

def visualize_validation_sample(
        image: torch.Tensor,
        boxes: torch.Tensor,
        pred_html: str,
        true_html: str,
        pred_otsl: str,
        true_otsl: str,
        pointer_logits: Optional[torch.Tensor],
        step: int,
        loss_components: Dict[str, float],
        viz_dir: Path
    ) -> None:
        """검증 샘플 시각화"""
        plt.figure(figsize=(20, 8))
        
        # 이미지 역정규화
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = image.cpu().numpy()
        for i in range(3):
            img[i] = img[i] * std[i].item() + mean[i].item()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        H, W = image.shape[1:]
        
        # OTSL 토큰 파싱
        true_otsl_tokens = true_otsl.split()
        pred_otsl_tokens = pred_otsl.split()
        
        # 1-1. GT 박스와 OTSL 태그 시각화 (왼쪽)
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img)
        ax1.set_title('Ground Truth Boxes with OTSL Tags')
        
        # GT OTSL 태그에서 각 박스의 태그 찾기
        box_tags = []
        current_tag = None
        tag_count = 0
        
        for token in true_otsl_tokens:
            if token == 'NL':
                continue
            if tag_count < len(boxes):
                box_tags.append(token)
                tag_count += 1
        
        for i, (box, tag) in enumerate(zip(boxes, box_tags)):
            x1, y1, x2, y2 = box.cpu().numpy()
            color = 'r' if tag == 'C' else 'b' if tag == 'L' else 'g' if tag == 'U' else 'y'
            
            rect = patches.Rectangle(
                (x1 * W, y1 * H),
                (x2 - x1) * W,
                (y2 - y1) * H,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            ax1.add_patch(rect)
            
            # 박스 번호와 OTSL 태그 표시
            ax1.text(x1 * W, y1 * H, 
                    f"Box {i}\n{tag}", 
                    color='black', fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # 1-2. Pred 박스와 OTSL 태그 시각화 (오른쪽)
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(img)
        ax2.set_title('Predicted Box-OTSL Tag Matching')
        
        if pointer_logits is not None:
            pointer_logits = pointer_logits[0]  # 첫 번째 배치만 사용
            max_indices = pointer_logits.argmax(dim=-1)
            max_values = pointer_logits.max(dim=-1)[0]
            
            # confidence에 따른 색상 맵 생성
            cmap = plt.cm.get_cmap('YlOrRd')
            
            for i, box in enumerate(boxes):
                if max_indices[i].item() > 0:  # 0은 패딩
                    x1, y1, x2, y2 = box.cpu().numpy()
                    confidence = torch.sigmoid(max_values[i]).item()
                    
                    # 예측된 OTSL 태그 가져오기
                    token_idx = max_indices[i].item()
                    if token_idx < len(pred_otsl_tokens):
                        otsl_tag = pred_otsl_tokens[token_idx]
                    else:
                        otsl_tag = 'UNK'
                    
                    color = 'r' if otsl_tag == 'C' else 'b' if otsl_tag == 'L' else 'g' if otsl_tag == 'U' else 'y'
                    
                    rect = patches.Rectangle(
                        (x1 * W, y1 * H),
                        (x2 - x1) * W,
                        (y2 - y1) * H,
                        linewidth=2,
                        edgecolor=color,
                        facecolor=color,
                        alpha=confidence * 0.3
                    )
                    ax2.add_patch(rect)
                    
                    # 박스 번호, OTSL 태그, confidence 표시
                    ax2.text(x1 * W, y1 * H, 
                            f"Box {i}\n{otsl_tag}\n{confidence:.2f}", 
                            color='black', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Loss components 출력
        loss_str = " | ".join([
            f"{k.replace('_', ' ').title()}: {v:.4f}"
            for k, v in loss_components.items()
        ])
        plt.figtext(0.1, 0.02, loss_str, fontsize=10)
        
        plt.figtext(0.1, -0.08,
                    f"Pred OTSL: {pred_otsl[:100]}...\n"
                    f"True OTSL: {true_otsl[:100]}...\n"
                    f"Pred HTML: {pred_html[:100]}...\n"
                    f"True HTML: {true_html[:100]}...",
                    fontsize=8)
        
        plt.tight_layout()
        
        # 저장
        img_dir = viz_dir / "images"
        txt_dir = viz_dir / "txts"   
        img_dir.mkdir(exist_ok=True)
        txt_dir.mkdir(exist_ok=True)
        save_path = img_dir / f'val_step_{step:08d}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
        plt.close()
        save_txt = txt_dir / f'val_step_{step:08d}.txt'
        with open(save_txt, 'w') as f:
            f.write(f"Pred OTSL: {pred_otsl}\n")
            f.write(f"True OTSL: {true_otsl}\n")
            f.write(f"Pred HTML: {pred_html}\n")
            f.write(f"True HTML: {true_html}")