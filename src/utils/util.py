import os
import wandb
from datetime import datetime
from typing import Dict, Tuple, List
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
    """HTML 구조를 OTSL 토큰으로 변환"""
    html_structure = ann['html']['structure']
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
    
    # 2. 그리드 초기화
    grid = [[None] * num_cols for _ in range(num_rows)]
    
    # 3. HTML 토큰을 순회하며 OTSL 그리드 생성
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
            
            # 이미 채워진 셀 건너뛰기
            while current_col < num_cols and grid[current_row][current_col] is not None:
                current_col += 1
                
            if current_col < num_cols:
                # 1. 시작 셀은 항상 'C'
                grid[current_row][current_col] = 'C'
                
                # 2. colspan 처리 (수평 병합)
                for c in range(current_col + 1, min(current_col + colspan, num_cols)):
                    grid[current_row][c] = 'L'
                
                # 3. rowspan 처리 (수직 병합)
                if rowspan > 1:
                    for r in range(current_row + 1, min(current_row + rowspan, num_rows)):
                        if colspan == 1:
                            grid[r][current_col] = 'U'
                        else:
                            grid[r][current_col] = 'U'
                            for c in range(current_col + 1, min(current_col + colspan, num_cols)):
                                grid[r][c] = 'X'
                
                current_col += colspan
        
        # rowspan 처리를 위한 추가 검사
        elif token == '</tr>':
            # 현재 행의 나머지 열들에 대해 위쪽 셀의 rowspan 확인
            while current_col < num_cols:
                if grid[current_row][current_col] is None and current_row > 0:
                    # 위쪽 셀 확인
                    above_cell = grid[current_row - 1][current_col]
                    if above_cell == 'C' or above_cell == 'U':
                        grid[current_row][current_col] = 'U'
                        # 위쪽 셀이 colspan된 경우 'X' 처리
                        next_col = current_col + 1
                        while next_col < num_cols and grid[current_row - 1][next_col] == 'L':
                            grid[current_row][next_col] = 'X'
                            next_col += 1
                current_col += 1
        i += 1
    
    # 4. 빈 셀을 'C'로 채우기
    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] is None:
                grid[i][j] = 'C'
    
    # 5. OTSL 시퀀스 생성
    tokens = []
    for row in grid:
        tokens.extend(row)
        tokens.append('NL')
    
    otsl_sequence = ' '.join(tokens)
    
    # # 디버깅을 위한 출력
    # print("\nOTSL Grid (by rows):")
    # print(f"Dimensions: {num_rows} rows x {num_cols} columns")
    # rows = otsl_sequence.split('NL')
    # for row in rows[:-1]:  # 마지막 빈 행 제외
    #     print(row.strip())
    
    return otsl_sequence


class CustomJSONEncoder(json.JSONEncoder):
    """텐서와 넘파이 배열을 JSON으로 직렬화하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        return super().default(obj)
    