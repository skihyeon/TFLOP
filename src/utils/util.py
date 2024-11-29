import os
import wandb
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import torch
import json

def init_wandb(model_config, train_config):
        """wandb 초기화"""
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(
            entity= os.environ.get('WANDB_ENTITY'),
            project="TFLOP",
            name=datetime.now().strftime('%Y%m%d_%H%M%S'),
            config={
                "model_config": model_config.__dict__,
                "train_config": train_config.__dict__
            },
            encoder=CustomJSONEncoder
        )
def extract_spans_from_html(html_structure: Dict) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """HTML 구조에서 span 정보 추출"""
    tokens = html_structure['tokens']
    
    # 1. 그리드 크기 계산
    num_rows = 0
    num_cols = 0
    max_cols = 0
    
    i = 0
    in_first_row = False
    while i < len(tokens):
        token = tokens[i]
        if token == '<tr>':
            num_rows += 1
            current_cols = 0
            in_first_row = (num_rows == 1)
        elif (token == '<td' or token == '<td>') and in_first_row:
            # colspan 확인
            colspan = 1
            if token == '<td':
                i += 1
                while i < len(tokens) and tokens[i] != '>':
                    if 'colspan="' in tokens[i]:
                        colspan = int(tokens[i].split('"')[1])
                    i += 1
            current_cols += colspan
            max_cols = max(max_cols, current_cols)
        elif token == '</tr>' and in_first_row:
            num_cols = max_cols
            in_first_row = False
        i += 1
    
    # 2. Span 행렬 초기화
    row_span_matrix = np.ones((num_rows, num_cols))
    col_span_matrix = np.ones((num_rows, num_cols))
    
    # 3. 셀 정보 추출 및 span 행렬 생성
    processed_cells = []
    current_row = -1
    current_col = 0
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        if token == '<tr>':
            current_row += 1
            current_col = 0
        elif token == '<td' or token == '<td>':
            # colspan과 rowspan 확인
            colspan = 1
            rowspan = 1
            
            if token == '<td':
                i += 1
                while i < len(tokens) and tokens[i] != '>':
                    if 'colspan="' in tokens[i]:
                        colspan = int(tokens[i].split('"')[1])
                    elif 'rowspan="' in tokens[i]:
                        rowspan = int(tokens[i].split('"')[1])
                    i += 1
            
            # 이미 처리된 셀 건너뛰기
            while current_col < num_cols and (
                row_span_matrix[current_row, current_col] == 0 or 
                col_span_matrix[current_row, current_col] == 0
            ):
                current_col += 1
                
            if current_col < num_cols:
                # 셀 정보 저장
                cell = {
                    'row_idx': current_row,
                    'col_idx': current_col,
                    'rowspan': rowspan,
                    'colspan': colspan
                }
                processed_cells.append(cell)
                
                # Span 정보 저장
                if colspan > 1 or rowspan > 1:
                    row_span_matrix[current_row, current_col] = rowspan
                    col_span_matrix[current_row, current_col] = colspan
                    
                    # span된 셀들은 0으로 마킹
                    for r in range(current_row, min(current_row + rowspan, num_rows)):
                        for c in range(current_col, min(current_col + colspan, num_cols)):
                            if r != current_row or c != current_col:
                                row_span_matrix[r, c] = 0
                                col_span_matrix[r, c] = 0
                
                current_col += colspan
        i += 1
    
    return processed_cells, row_span_matrix, col_span_matrix
    
def convert_html_to_otsl(ann: Dict) -> str:
    try:
        html_tokens = ann['html']['structure']['tokens']
        
        # 1. 그리드 크기와 span 정보 수집
        num_cols = 0
        current_row = 0
        current_col = 0
        spans = {}
        
        i = 0
        while i < len(html_tokens):
            token = html_tokens[i]
            
            if token == '<tr>':
                current_col = 0
                
            elif token == '</tr>':
                current_row += 1
                num_cols = max(num_cols, current_col)
                
            elif token == '<td' or token == '<td>':
                colspan = 1
                rowspan = 1
                
                if token == '<td':
                    i += 1
                    while i < len(html_tokens) and html_tokens[i] != '>':
                        if 'colspan=' in html_tokens[i]:
                            colspan = int(html_tokens[i].split('"')[1])
                        elif 'rowspan=' in html_tokens[i]:
                            rowspan = int(html_tokens[i].split('"')[1])
                        i += 1
                
                spans[(current_row, current_col)] = (rowspan, colspan)
                current_col += colspan
            
            i += 1
        
        num_rows = current_row + 1
        
        if num_rows == 0 or num_cols == 0:
            raise ValueError(f"Invalid table dimensions: rows={num_rows}, cols={num_cols}")
        
        
        grid = [[None] * num_cols for _ in range(num_rows)]

        # 1단계: rowspan이 있는 셀들 먼저 처리
        for (row, col), (rowspan, colspan) in spans.items():
            if rowspan > 1:
                # 시작 셀
                grid[row][col] = 'C'
                # 가로 방향 L 처리
                for c in range(col + 1, col + colspan):
                    grid[row][c] = 'L'
                # 아래 방향 U 처리
                for r in range(row + 1, row + rowspan):
                    for c in range(col, col + colspan):  # col부터 시작 (수정)
                        grid[r][c] = 'U'

        # 2단계: 각 행의 colspan 처리
        for row in range(num_rows):
            # 각 행에서 왼쪽부터 처리
            col = 0
            while col < num_cols:
                if grid[row][col] == 'U':  # U로 이미 설정된 셀은 건너뛰기
                    col += 1
                    continue
                    
                # 현재 위치의 span 정보 찾기
                span_info = next((span for (r, c), span in spans.items() 
                                if r == row and c == col), None)
                
                if span_info and span_info[1] > 1:  # colspan이 있는 경우
                    grid[row][col] = 'C'
                    for c in range(col + 1, col + span_info[1]):
                        if grid[row][c] != 'U':  # U가 아닌 경우만 L로 설정
                            grid[row][c] = 'L'
                    col += span_info[1]
                else:
                    col += 1
                        
       # 4. X 규칙 적용
        for r in range(1, num_rows):
            for c in range(1, num_cols):
                if grid[r][c] in ['U']:  # U 셀이고 첫 행/열이 아님
                    left_neighbor = grid[r][c-1]
                    upper_neighbor = grid[r-1][c]
                        
                    # X 변환 조건 체크
                    if left_neighbor in ['U', 'X'] and upper_neighbor in ['L', 'X']:
                        grid[r][c] = 'X'
        
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is None:
                    grid[r][c] = 'C'
        
        # 5. OTSL 시퀀스 생성
        otsl_tokens = []
        for row in grid:
            otsl_tokens.extend(row)
            otsl_tokens.append('NL')
        
        return " ".join(otsl_tokens)
        
    except Exception as e:
        # print(f"Error converting HTML to OTSL: {str(e)}")
        # print(f"HTML tokens: {html_tokens}")
        # raise ValueError("Invalid OTSL syntax")
        return None
    
def construct_table_html(
    otsl_sequence: str,
    text_regions: List[Dict[str, Union[str, List[float]]]],
    pointer_logits: Optional[torch.Tensor] = None,
    confidence_threshold: float = 0.5  # 포인터 매칭 신뢰도 임계값
) -> str:
    """OTSL 시퀀스와 text region 정보를 결합하여 완전한 HTML 테이블 생성"""
    try:
        # 1. 입력 유효성 검사
        if not text_regions or not isinstance(text_regions, list):
            return f"<table><tr><td>Invalid text regions</td></tr></table>"
            
        # 1. OTSL 토큰을 파싱하여 그리드 구조 생성
        tokens = otsl_sequence.split()
        rows = []
        current_row = []
        
        for token in tokens:
            if token == 'NL':
                if current_row:  # 빈 행 제외
                    rows.append(current_row)
                    current_row = []
            else:
                current_row.append(token)
        if current_row:  # 마지막 행 처리
            rows.append(current_row)
        
        # 2. 그리드 구조 검증
        if not rows:
            raise ValueError("Invalid OTSL sequence: No valid rows found")
        num_cols = len(rows[0])
        if not all(len(row) == num_cols for row in rows):
            raise ValueError("Invalid OTSL sequence: Inconsistent row lengths")
        
        # 3. span 정보 추적을 위한 그리드 생성
        num_rows = len(rows)
        cell_grid = [[None] * num_cols for _ in range(num_rows)]
        data_tag_positions = []  # (row, col, html_pos) 튜플 저장
        
        # 4. HTML 생성 및 span 정보 추적
        html = ["<table>"]
        html_pos = 1  # <table> 다음부터 시작
        
        for i, row in enumerate(rows):
            html.append("<tr>")
            html_pos += 1
            
            j = 0
            while j < len(row):
                if cell_grid[i][j] is not None:
                    j += 1
                    continue
                    
                token = row[j]
                colspan = 1
                rowspan = 1
                
                # Span 정보 계산
                if token == 'L' or token == 'X':
                    colspan = 2
                if token == 'U' or token == 'X':
                    rowspan = 2
                
                # Cell 태그 생성
                cell_tag = "<td"
                if colspan > 1:
                    cell_tag += f' colspan="{colspan}"'
                if rowspan > 1:
                    cell_tag += f' rowspan="{rowspan}"'
                cell_tag += "></td>"
                
                # Span 정보 그리드에 표시
                for ri in range(i, min(i + rowspan, num_rows)):
                    for ci in range(j, min(j + colspan, num_cols)):
                        cell_grid[ri][ci] = (i, j)  # 원본 셀 위치 저장
                
                # 데이터 셀('C') 위치 저장
                if token == 'C':
                    data_tag_positions.append((i, j, html_pos))
                
                html.append(cell_tag)
                html_pos += 1
                j += colspan
            
            html.append("</tr>")
            html_pos += 1
        
        html.append("</table>")
        
        # 5. text region 매핑
        if pointer_logits is not None:
            # 기존 pointer logits 처리 코드
            try:
                pointer_probs = torch.softmax(pointer_logits, dim=-1)
                box_to_cell = {}
                for box_idx in range(min(pointer_logits.size(0), len(text_regions))):
                    probs = pointer_probs[box_idx]
                    max_prob, cell_idx = probs.max(dim=0)
                    
                    if (max_prob.item() >= confidence_threshold and 
                        cell_idx < len(data_tag_positions)):
                        row, col, html_pos = data_tag_positions[cell_idx]
                        if html_pos not in box_to_cell:
                            box_to_cell[html_pos] = []
                        box_to_cell[html_pos].append({
                            'box_idx': box_idx,
                            'confidence': max_prob.item()
                        })
            except Exception as e:
                print(f"Warning: Failed to process pointer logits: {str(e)}")
                box_to_cell = {}
        else:
            # GT의 경우: text regions를 순서대로 data cell positions에 매핑
            box_to_cell = {}
            for box_idx, (row, col, html_pos) in enumerate(data_tag_positions):
                if box_idx < len(text_regions):
                    if html_pos not in box_to_cell:
                        box_to_cell[html_pos] = []
                    box_to_cell[html_pos].append({
                        'box_idx': box_idx,
                        'confidence': 1.0
                    })
        
        # HTML에 text 삽입
        html_with_text = []
        for i, tag in enumerate(html):
            if i in box_to_cell:
                try:
                    # 해당 셀에 매핑된 모든 text region의 텍스트 결합
                    cell_texts = []
                    for box in box_to_cell[i]:
                        if (box['box_idx'] < len(text_regions) and
                            'text' in text_regions[box['box_idx']]):
                            text = text_regions[box['box_idx']]['text'].strip()
                            if text:  # 빈 텍스트 제외
                                cell_texts.append(text)
                    
                    cell_text = ' '.join(cell_texts) if cell_texts else ''
                    
                    # </td> 직전에 텍스트 삽입
                    if '</td>' in tag:
                        tag_parts = tag.split('</td>')
                        html_with_text.append(f"{tag_parts[0]}{cell_text}</td>")
                    else:
                        html_with_text.append(tag)
                except Exception as e:
                    html_with_text.append(tag)
            else:
                html_with_text.append(tag)
        
        return '\n'.join(html_with_text)
        
    except Exception as e:
        return f"<table><tr><td>Error: {str(e)}</td></tr></table>"
    

class CustomJSONEncoder(json.JSONEncoder):
    """텐서와 넘파이 배열을 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        return super().default(obj)
    