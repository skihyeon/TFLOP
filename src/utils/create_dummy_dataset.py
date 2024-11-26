import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw as ImageDrawType
from PIL.ImageFont import ImageFont as ImageFontType
from typing import Dict, List, Tuple
import textwrap

def create_dummy_table_structure(num_rows: int, num_cols: int) -> Tuple[List[Dict], str, List[int]]:
    """논문 스타일의 테이블 구조 생성"""
    
    # 1. 계층적 구조 생성 (Figure 4 참조)
    spans = np.ones((num_rows, num_cols, 2), dtype=int)  # [row_span, col_span]
    cell_used = np.zeros((num_rows, num_cols), dtype=bool)
    
    # 2. 복잡한 span 구조 생성 (50% 확률로)
    if np.random.random() < 0.5:
        # 계층적 rowspan 또는 colspan 생성
        for row in range(num_rows):
            for col in range(num_cols):
                if cell_used[row, col]:
                    continue
                    
                if np.random.random() < 0.6:  # 60% 확률로 span 생성
                    span_type = np.random.choice(['row', 'col'])
                    if span_type == 'row' and row < num_rows - 1:
                        span_size = min(np.random.randint(2, 5), num_rows - row)
                        spans[row, col, 0] = span_size
                        cell_used[row:row+span_size, col] = True
                    elif span_type == 'col' and col < num_cols - 1:
                        span_size = min(np.random.randint(2, 5), num_cols - col)
                        spans[row, col, 1] = span_size
                        cell_used[row, col:col+span_size] = True
    
    # 3. Box-Tag 관계 및 셀 정보 생성
    cells = []
    html_parts = ["<table>"]
    td_positions = []  # td 태그의 실제 위치 저장
    current_position = 1  # <table> 태그 다음부터 시작
    
    for row in range(num_rows):
        html_parts.append("<tr>")
        current_position += 1
        
        for col in range(num_cols):
            if cell_used[row, col] and not (spans[row, col] > 1).any():
                continue
            
            # 1. 셀 타입 결정 (header vs data)
            is_header = (row == 0 or col == 0)
            tag = "th" if is_header else "td"
            
            # 2. Span 속성 추가
            span_attrs = []
            if spans[row, col, 0] > 1:
                span_attrs.append(f'rowspan="{spans[row, col, 0]}"')
            if spans[row, col, 1] > 1:
                span_attrs.append(f'colspan="{spans[row, col, 1]}"')
            
            # 3. Empty cell 생성 (10% 확률)
            is_empty = np.random.random() < 0.1
            text = "" if is_empty else (
                generate_header_text() if is_header else generate_scientific_text()
            )
            
            # 4. Many-to-one 관계 생성 (20% 확률)
            is_many_to_one = np.random.random() < 0.2
            num_boxes = np.random.randint(2, 4) if is_many_to_one else 1
            
            span_str = ' ' + ' '.join(span_attrs) if span_attrs else ''
            tag_position = current_position
            
            if tag == "td":
                # Many-to-one 관계를 위한 여러 box indices
                td_indices = []
                for _ in range(num_boxes):
                    td_pos = tag_position + row * num_cols + col + np.random.randint(-2, 3)
                    td_indices.append(td_pos)
                td_positions.extend(td_indices)
            
            html_parts.append(f"<{tag}{span_str}>{text}</{tag}>")
            current_position += 1
            
            # 5. Overlap 정보 계산
            row_start = row
            row_end = row + spans[row, col, 0]
            col_start = col
            col_end = col + spans[row, col, 1]
            
            row_overlap = list(range(row_start, row_end))
            col_overlap = list(range(col_start, col_end))
            
            # 6. Box 정보 저장
            cells.append({
                'text': text,
                'row': row,
                'col': col,
                'row_span': spans[row, col, 0],
                'col_span': spans[row, col, 1],
                'tag': tag,
                'tag_index': tag_position,
                'td_indices': td_indices if tag == "td" and is_many_to_one else [tag_position],
                'is_empty': is_empty,
                'is_td': tag == "td",
                'overlaps': {
                    'row': row_overlap,
                    'col': col_overlap
                }
            })
        
        html_parts.append("</tr>")
        current_position += 1
    
    html_parts.append("</table>")
    html_structure = " ".join(html_parts)
    
    return cells, html_structure, sorted(list(set(td_positions)))

def generate_header_text() -> str:
    """과학 논문 스타일의 헤더 텍스트 생성"""
    headers = [
        "Model", "Architecture", "F1-Score", "mAP", "Loss",
        "Epochs", "Batch Size", "Learning Rate", "Optimizer",
        "Ablation", "Baseline", "Ours", "Delta (%)", 
        "Train", "Valid", "Test", "Average", "Std",
        "Method Type", "Dataset Type", "Metrics",
        "Performance", "Settings", "Results"
    ]
    return np.random.choice(headers)

def generate_scientific_text() -> str:
    """과학 논문 스타일의 데이터 텍스트 생성"""
    text_types = [
        # 수치형 데이터
        lambda: f"{np.random.random():.4f}",
        lambda: f"{np.random.normal(0, 1):.2e}",
        lambda: f"{np.random.randint(80, 100)}.{np.random.randint(0, 100):02d}%",
        # 측정값
        lambda: f"{np.random.randint(1, 1000)} ± {np.random.randint(1, 100)}",
        lambda: f"{np.random.random()*1e-6:.2e} mol/L",
        # 통계값
        lambda: f"p < {10**-np.random.randint(1, 5):.0e}",
        lambda: f"r = {np.random.random():.3f}",
        # 범주형 데이터
        lambda: np.random.choice(["Control", "Treatment A", "Treatment B"]),
        lambda: f"Group {np.random.randint(1, 5)}",
        # 모델 성능 관련
        lambda: f"{np.random.randint(85, 100)}.{np.random.randint(0, 99):02d}",
        lambda: f"{np.random.random():.3f} ± {np.random.random():.3f}",
        # 학습 파라미터
        lambda: f"{2**np.random.randint(4, 8)}",  # batch sizes
        lambda: f"{10**-np.random.randint(2, 5):.0e}",  # learning rates
        # 실험 결과
        lambda: f"+{np.random.randint(1, 10)}.{np.random.randint(0, 99):02d}%",
        lambda: f"-{np.random.randint(1, 10)}.{np.random.randint(0, 99):02d}%",
        # 빈 셀도 생성 (논문의 Layout Pointer 부분 참고)
        lambda: ""
    ]
    # 빈 셀 생성 확률 추가 (10%)
    if np.random.random() < 0.1:
        return ""
    return np.random.choice(text_types)()

def fit_text_to_cell(
    draw: ImageDrawType, 
    text: str, 
    max_width: int, 
    max_height: int, 
    font: ImageFontType
) -> Tuple[str, ImageFontType]:
    """텍스트를 셀 크기에 맞게 조정"""
    # 초기 텍스트 크기 측정
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 텍스트가 너무 길면 줄바꿈
    if text_width > max_width:
        # 셀 너비를 기준으로 한 글자당 평균 너비 계산
        avg_char_width = text_width / len(text)
        chars_per_line = int(max_width / avg_char_width)
        # 텍스트 줄바꿈
        text = textwrap.fill(text, width=max(1, chars_per_line-1))
        
    # 줄바꿈 후 텍스트 크기 다시 측정
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 텍스트가 여전히 셀보다 크면 폰트 크기 조정
    current_font_size = font.size
    while (text_width > max_width * 0.9 or text_height > max_height * 0.9) and current_font_size > 8:
        current_font_size -= 2
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", current_font_size)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    
    return text, font

def create_dummy_dataset(
    output_dir: str,
    num_train: int = 500,  # 논문 4.1: PubTabNet 크기
    num_val: int = 100
) -> None:
    """더미 데이터셋 생성"""
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # 학습/검증 데이터 생성
    for split, num_samples in [('train', num_train), ('val', num_val)]:
        dataset = {}
        
        for idx in range(num_samples):
            # 테이블 크기 범위 확대 (2x2~10x10)
            num_rows = np.random.randint(2, 11)
            num_cols = np.random.randint(2, 11)
            
            # 테이블 구조 생성
            cells, html_structure, td_positions = create_dummy_table_structure(num_rows, num_cols)
            
            # 이미지 생성 (768x768 고정 - 논문 4.2)
            image = Image.new('RGB', (768, 768), (250, 250, 250))
            draw = ImageDraw.Draw(image)
            
            # 폰트 설정
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # 셀 크기 계산
            cell_width = 768 // num_cols
            cell_height = 768 // num_rows
            
            # 바운딩 박스 정보 저장
            boxes = []
            
            # 테이블 그리기
            for cell in cells:
                # 셀 좌표 계산
                x1 = cell['col'] * cell_width
                y1 = cell['row'] * cell_height
                x2 = x1 + cell_width * cell['col_span']
                y2 = y1 + cell_height * cell['row_span']
                
                # 셀 테두리 그리기
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
                
                # 텍스트 추가 (empty가 아닌 경우만)
                if not cell['is_empty']:
                    text = cell['text']
                    text, adjusted_font = fit_text_to_cell(
                        draw, text, x2-x1, y2-y1, font
                    )
                    
                    # 텍스트 중앙 정렬
                    text_bbox = draw.textbbox((0, 0), text, font=adjusted_font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = x1 + (cell_width - text_width) // 2
                    text_y = y1 + (cell_height - text_height) // 2
                    
                    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=adjusted_font)
                
                # 바운딩 박스 정보 저장 (정규화된 좌표)
                boxes.append({
                    'text': cell['text'],
                    'bbox': [float(x1/768), float(y1/768), float(x2/768), float(y2/768)],
                    'row_span': int(cell['row_span']),
                    'col_span': int(cell['col_span']),
                    'tag': cell['tag'],
                    'tag_indices': cell['td_indices'],
                    'is_td': cell['is_td'],
                    'is_empty': cell['is_empty'],
                    'overlaps': cell['overlaps']
                })
            
            # 이미지 저장
            image_id = f'{split}_{idx}'
            image_path = os.path.join(output_dir, 'images', f'{image_id}.png')
            image.save(image_path)
            
            # 샘플 정보 저장
            dataset[image_id] = {
                'boxes': boxes,
                'html': html_structure,
                'td_positions': td_positions,
                'split': split
            }
        
        # 어노테이션 저장
        annotation_path = os.path.join(output_dir, 'annotations', f'{split}.json')
        with open(annotation_path, 'w') as f:
            json.dump(dataset, f, indent=2)

def main():
    """더미 데이터셋 생성 실행"""
    output_dir = 'data/dummy_pubtabnet'
    create_dummy_dataset(output_dir)
    print("Dummy dataset created successfully!")

if __name__ == '__main__':
    main()