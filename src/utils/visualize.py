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
    
    # OTSL 토큰을 그리드로 변환
    def tokens_to_grid(otsl_str):
        tokens = [t for t in otsl_str.split() if t not in ['[BOS]', '[EOS]', '[PAD]']]
        grid = []
        current_row = []
        for token in tokens:
            if token == 'NL':
                if current_row:
                    grid.append(current_row)
                    current_row = []
            else:
                current_row.append(token)
        if current_row:
            grid.append(current_row)
        return grid
    
    true_grid = tokens_to_grid(true_otsl)
    pred_grid = tokens_to_grid(pred_otsl)
    
    def visualize_grid(ax, grid, boxes, pointer_logits=None, title=''):
        ax.imshow(img)
        ax.set_title(title)
        
        # bbox 인덱스 추적
        box_idx = 0
        cell_boxes = {}  # (i,j) -> box 매핑 저장
        
        # 1. 먼저 모든 'C' 토큰의 bbox 그리기
        for i, row in enumerate(grid):
            for j, token in enumerate(row):
                if token == 'C':
                    # bbox가 있는 경우
                    if box_idx < len(boxes):
                        box = boxes[box_idx].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        rect = patches.Rectangle(
                            (x1 * W, y1 * H),
                            (x2 - x1) * W,
                            (y2 - y1) * H,
                            linewidth=2,
                            edgecolor='red',
                            facecolor='none',
                            alpha=0.7
                        )
                        ax.add_patch(rect)
                        
                        # 셀 정보 표시
                        text = f"({i},{j})\n{token}"
                        if pointer_logits is not None:
                            confidence = torch.sigmoid(pointer_logits[0, box_idx].max()).item()
                            text += f"\n{confidence:.2f}"
                        
                        ax.text(x1 * W, y1 * H, 
                                text,
                                color='black', fontsize=5, 
                                bbox=dict(facecolor='white', alpha=0.7, pad=0.2))
                        
                        cell_boxes[(i,j)] = box
                        box_idx += 1
                    else:
                        # bbox가 없는 'C' 토큰 (빈 셀)
                        # 이전 셀의 위치를 기반으로 추정된 위치에 점선으로 표시
                        if j > 0 and (i,j-1) in cell_boxes:
                            prev_box = cell_boxes[(i,j-1)]
                            x2, y1 = prev_box[2], prev_box[1]
                            ax.text(x2 * W + 10, y1 * H, 
                                    f"({i},{j})\n{token}\n(empty)",
                                    color='black', fontsize=5,
                                    bbox=dict(facecolor='lightgray', alpha=0.3, pad=0.2))
                
                # 2. span 관계 표시 (L, U, X 토큰)
                elif token in ['L', 'U', 'X']:
                    # span 관계 찾기
                    ref_cells = []
                    if token in ['L', 'X'] and j > 0:  # 왼쪽 참조
                        ref_cells.append((i, j-1))
                    if token in ['U', 'X'] and i > 0:  # 위쪽 참조
                        ref_cells.append((i-1, j))
                    
                    # 참조하는 셀들과 화살표로 연결
                    for ref_i, ref_j in ref_cells:
                        if (ref_i, ref_j) in cell_boxes:
                            ref_box = cell_boxes[(ref_i, ref_j)]
                            x1, y1, x2, y2 = ref_box
                            
                            # 화살표 스타일 설정
                            arrow_style = patches.ArrowStyle('->', head_length=10, head_width=7)
                            color = 'blue' if token == 'L' else 'green' if token == 'U' else 'orange'
                            
                            # 화살표 그리기
                            arrow = patches.FancyArrowPatch(
                                (x2 * W, y1 * H),  # 시작점
                                (x2 * W + 30, y1 * H),  # 끝점
                                arrowstyle=arrow_style,
                                color=color,
                                alpha=0.7
                            )
                            ax.add_patch(arrow)
                            
                            # span 정보 표시
                            ax.text(x2 * W + 35, y1 * H, 
                                    f"({i},{j})\n{token}", 
                                    color='black', fontsize=5,
                                    bbox=dict(facecolor=color, alpha=0.3, pad=0.2))
    
    # Ground Truth 시각화
    ax1 = plt.subplot(1, 2, 1)
    visualize_grid(ax1, tokens_to_grid(true_otsl), boxes, title='Ground Truth Structure')
    
    # Prediction 시각화
    ax2 = plt.subplot(1, 2, 2)
    visualize_grid(ax2, tokens_to_grid(pred_otsl), boxes, pointer_logits, title='Predicted Structure')
    
    
    # 저장
    img_dir = viz_dir / "images"
    txt_dir = viz_dir / "txts"   
    img_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)
    
    save_path = img_dir / f'val_step_{step:08d}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.close()
    
    # OTSL 및 HTML 저장
    save_txt = txt_dir / f'val_step_{step:08d}.txt'
    with open(save_txt, 'w') as f:
        f.write(f"Ground Truth Grid Structure:\n")
        for i, row in enumerate(true_grid):
            f.write(f"Row {i}: {' '.join(row)}\n")
        f.write(f"\nPredicted Grid Structure:\n")
        for i, row in enumerate(pred_grid):
            f.write(f"Row {i}: {' '.join(row)}\n")
        f.write(f"\nPred OTSL: {pred_otsl}\n")
        f.write(f"True OTSL: {true_otsl}\n")
        f.write(f"Pred HTML: {pred_html}\n")
        f.write(f"True HTML: {true_html}")