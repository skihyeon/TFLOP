import os
import wandb
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import torch
import json
import itertools
from models.otsl_tokenizer import OTSLTokenizer

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
    
    # 2. Span 행렬 초기화 - 각 셀의 실제 span 값을 저장
    row_span_matrix = np.zeros((num_rows, num_cols))  # rowspan 값 저장
    col_span_matrix = np.zeros((num_rows, num_cols))  # colspan 값 저장
    
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
            colspan = 1
            rowspan = 1
            
            # colspan과 rowspan 값 추출
            if token == '<td':
                i += 1
                while i < len(tokens) and tokens[i] != '>':
                    if 'colspan="' in tokens[i]:
                        colspan = int(tokens[i].split('"')[1])
                    elif 'rowspan="' in tokens[i]:
                        rowspan = int(tokens[i].split('"')[1])
                    i += 1
            
            # 현재 위치 찾기
            while current_col < num_cols and (
                row_span_matrix[current_row, current_col] > 0 or 
                col_span_matrix[current_row, current_col] > 0
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
                
                # Span 정보 저장 - 실제 span 값 사용
                for r in range(current_row, min(current_row + rowspan, num_rows)):
                    for c in range(current_col, min(current_col + colspan, num_cols)):
                        row_span_matrix[r, c] = rowspan
                        col_span_matrix[r, c] = colspan
                
                current_col += colspan
        i += 1
    
    return processed_cells, row_span_matrix, col_span_matrix

def convert_html_to_otsl(ann: Dict) -> Tuple[List[str], List[bool]]:
    """HTML 구조를 OTSL로 변환하고 데이터 존재 여부를 반환
    
    Args:
        ann: PubTabNet annotation 딕셔너리
        
    Returns:
        otsl_sequence: OTSL 토큰 시퀀스 (오류 시 None)
        has_data_1D_list: 각 셀의 데이터 존재 여부 (1D 리스트)
    """
    if 'html' not in ann or 'structure' not in ann['html'] or 'cells' not in ann['html']:
        print(f"Error: Missing required HTML structure")
        return None, []
        
    html_tokens = ann['html']['structure']['tokens']
    cells = ann['html']['cells']
    
    # cells의 데이터 존재 여부 미리 계산
    has_content = [len(cell['tokens']) > 0 for cell in cells]
    current_cell_idx = 0
    
    # 1. 그리드 크기와 span 정보 수집
    num_cols = 0
    current_row = -1
    current_col = 0
    spans = {}  # (row, col) -> (rowspan, colspan)
    active_rowspans = {}  # col -> (start_row, rowspan)
    
    i = 0

    while i < len(html_tokens):
        token = html_tokens[i]
        
        if token == '<tr>':
            current_row += 1
            current_col = 0
            # 현재 행에서 활성화된 rowspan 확인
            for col in range(num_cols):
                if col in active_rowspans:
                    start_row, rowspan = active_rowspans[col]
                    if current_row >= start_row + rowspan:
                        del active_rowspans[col]
            
        elif token == '</tr>':
            num_cols = max(num_cols, current_col)
            
        elif token == '<td' or token == '<td>':
            # 현재 위치가 활성 rowspan 아래에 있는지 확인
            while current_col in active_rowspans:
                start_row, rowspan = active_rowspans[current_col]
                if current_row < start_row + rowspan:
                    current_col += 1
                else:
                    del active_rowspans[current_col]
            
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
            
            # 현재 셀의 span 정보와 데이터 존재 여부 저장
            spans[(current_row, current_col)] = (rowspan, colspan, 
                has_content[current_cell_idx] if current_cell_idx < len(has_content) else False)
            current_cell_idx += 1
            
            if rowspan > 1:
                for c in range(current_col, current_col + colspan):
                    active_rowspans[c] = (current_row, rowspan)
            current_col += colspan
        
        i += 1
        
    num_rows = current_row + 1
    
    # 2. 그리드 생성 및 OTSL 토큰 채우기
    grid = [[None] * num_cols for _ in range(num_rows)]
    has_data = [[False] * num_cols for _ in range(num_rows)]
    
    # 그리드 생성 및 OTSL 토큰 채우기 부분 수정
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] is not None:
                continue
                
            # 현재 셀이 어떤 span의 영향을 받는지 확인
            affecting_spans = []  # [(row, col, rowspan, colspan), ...]
            
            # 위쪽 셀들의 rowspan 확인
            if row > 0:
                for r in range(row-1, -1, -1):
                    if (r, col) in spans:
                        rowspan, colspan, _ = spans[(r, col)]
                        if row < r + rowspan:
                            affecting_spans.append((r, col, rowspan, colspan))
                            break

            # 왼쪽 셀들의 colspan 확인
            if col > 0:
                for c in range(col-1, -1, -1):
                    if (row, c) in spans:
                        rowspan, colspan, _ = spans[(row, c)]
                        if col < c + colspan:
                            affecting_spans.append((row, c, rowspan, colspan))
                            break
            
            # 태그 결정
            if len(affecting_spans) == 2:
                # 두 방향 모두에서 영향을 받는 경우
                grid[row][col] = 'X'
            elif len(affecting_spans) == 1:
                # 한 방향에서만 영향을 받는 경우
                span_row, span_col, rowspan, colspan = affecting_spans[0]
                if span_row != row:  # 위쪽 셀의 영향
                    grid[row][col] = 'U'
                else:  # 왼쪽 셀의 영향
                    grid[row][col] = 'L'
            elif (row, col) in spans:
                # 새로운 span의 시작점
                grid[row][col] = 'C'
                rowspan, colspan, has_content = spans[(row, col)]
                has_data[row][col] = has_content
                
                # span 영역 채우기
                for r in range(row, row + rowspan):
                    for c in range(col, col + colspan):
                        if r == row and c == col:
                            continue  # 시작점은 이미 처리됨
                        if r == row:
                            grid[r][c] = 'L'  # 같은 행의 span
                        elif c == col:
                            grid[r][c] = 'U'  # 같은 열의 span
                        else:
                            grid[r][c] = 'X'  # 대각선 방향의 span
            else:
                # 일반 셀
                grid[row][col] = 'C'
                has_data[row][col] = False
    
    # OTSL 시퀀스 생성
    otsl_tokens = []
    has_data_1D_list = []
   
    for row_idx, row in enumerate(grid):
        for col_idx, token in enumerate(row):
            if token is not None:
                otsl_tokens.append(token)
                # C 토큰이고 원본 셀인 경우에만 has_data 체크
                if token == 'C' and (row_idx, col_idx) in spans:
                    _, _, has_content = spans[(row_idx, col_idx)]
                    has_data_1D_list.append(has_content)
                else:
                    # C 토큰이 아니거나 원본 셀이 아닌 경우 항상 False
                    has_data_1D_list.append(False)
        
        # NL 토큰 추가 (마지막 행 포함)
        if row_idx < num_rows:
            otsl_tokens.append('NL')
            has_data_1D_list.append(False)
    
    # 길이 검증 (토큰 단위로)
    if len(otsl_tokens) != len(has_data_1D_list):
        print(f"Token length mismatch: tokens({len(otsl_tokens)}) != has_data({len(has_data_1D_list)})")
        return None, []
    
    # 토큰 리스트를 문자열로 변환하여 반환
    return otsl_tokens, has_data_1D_list

def construct_table_html_gt(html_data: Dict) -> str:
    """원본 HTML 구조를 사용하여 GT 테이블 생성"""
    try:
        html = ["<table>"]
        current_cell_idx = 0
        in_row = False
        
        # HTML 태그를 제거하는 함수
        def clean_text(tokens: List[str]) -> str:
            text = ''.join(tokens)
            # HTML 태그 제거
            text = text.replace('<b>', '').replace('</b>', '')
            text = text.replace('<i>', '').replace('</i>', '')
            text = text.replace('<sup>', '').replace('</sup>', '')
            text = text.replace('<sub>', '').replace('</sub>', '')
            return text
        
        for token in html_data['structure']['tokens']:
            if token == '<tr>':
                html.append("<tr>")
                in_row = True
            elif token == '</tr>':
                html.append("</tr>")
                in_row = False
            elif token == '</thead>' or token == '</tbody>':
                continue
            elif token == '<thead>' or token == '<tbody>':
                continue
            elif '<td' in token:
                cell_tag = "<td"
                if 'rowspan=' in token:
                    rowspan = token.split('rowspan="')[1].split('"')[0]
                    cell_tag += f' rowspan="{rowspan}"'
                if 'colspan=' in token:
                    colspan = token.split('colspan="')[1].split('"')[0]
                    cell_tag += f' colspan="{colspan}"'
                cell_tag += ">"
                
                # 셀 내용 추가 (HTML 태그 제거)
                if current_cell_idx < len(html_data['cells']):
                    cell_content = clean_text(html_data['cells'][current_cell_idx]['tokens'])
                    html.append(f"{cell_tag}{cell_content}</td>")
                    current_cell_idx += 1
                else:
                    html.append(f"{cell_tag}</td>")
            elif token == '</td>':
                continue
        
        html.append("</table>")
        return '\n'.join(html)
        
    except Exception as e:
        print(f"Error constructing GT HTML: {str(e)}")
        return f"<table><tr><td>Error: {str(e)}</td></tr></table>"

def construct_table_html_pred(
    otsl_sequence: str,
    text_regions: List[Dict[str, Union[str, List[float]]]],
    pointer_logits: torch.Tensor,
    confidence_threshold: float = 0.5
) -> str:
    """OTSL과 pointer logits를 사용하여 예측 테이블 생성"""
    try:
        # 1. OTSL 토큰을 그리드로 변환
        tokens = [t for t in otsl_sequence.split() if t not in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']]
        grid = []
        current_row = []
        
        # 2. 토큰 인덱스와 그리드 위치 매핑 (모든 태그 포함)
        token_positions = {}  # token_idx -> (row, col) 매핑
        token_idx = 0
        current_row_idx = 0
        
        for token in tokens:
            if token == 'NL':
                if current_row:
                    current_row.append('NL')
                    grid.append(current_row)
                    current_row = []
                current_row_idx += 1
                token_positions[token_idx] = (-1, -1)  # NL 태그 위치 표시
            else:
                current_row.append(token)
                token_positions[token_idx] = (current_row_idx, len(current_row) - 1)
            token_idx += 1
            
        if current_row:
            grid.append(current_row)

        # 3. 각 셀의 원본 셀(병합의 시작점) 찾기
        origin_cells = {}  # (row, col) -> (origin_row, origin_col)
        
        def find_origin_cell(row: int, col: int) -> Tuple[int, int]:
            if (row, col) in origin_cells:
                return origin_cells[(row, col)]
            
            token = grid[row][col]
            if token == 'C':
                origin_cells[(row, col)] = (row, col)
                return (row, col)
            
            if token == 'L':
                origin = find_origin_cell(row, col-1)
                origin_cells[(row, col)] = origin
                return origin
            
            if token == 'U':
                origin = find_origin_cell(row-1, col)
                origin_cells[(row, col)] = origin
                return origin
            
            if token == 'X':
                left_origin = find_origin_cell(row, col-1)
                up_origin = find_origin_cell(row-1, col)
                origin = min(left_origin, up_origin)
                origin_cells[(row, col)] = origin
                return origin
            
            return (row, col)

        # 4. 모든 셀의 원본 찾기
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if (i, j) not in origin_cells:
                    find_origin_cell(i, j)

        # 5. 각 원본 셀의 rowspan과 colspan 계산
        spans = {}  # (origin_row, origin_col) -> (rowspan, colspan)
        for origin in set(origin_cells.values()):
            spans[origin] = [1, 1]  # [rowspan, colspan]
        
        for (row, col), (origin_row, origin_col) in origin_cells.items():
            current_span = spans[(origin_row, origin_col)]
            current_span[0] = max(current_span[0], row - origin_row + 1)
            current_span[1] = max(current_span[1], col - origin_col + 1)

        # 6. pointer_logits를 사용하여 text_cells 매핑 생성
        text_cells = {}  # (row, col) -> text_idx 매핑
        pointer_probs = torch.softmax(pointer_logits, dim=1)
        
        for box_idx in range(pointer_probs.size(0)):
            max_prob, cell_idx = pointer_probs[box_idx].max(dim=0)
            if max_prob.item() >= confidence_threshold and cell_idx < len(token_positions):
                token_idx = cell_idx.item()
                if token_idx in token_positions:
                    row, col = token_positions[token_idx]
                    if row != -1:
                        text_cells[(row, col)] = box_idx

        # 7. HTML 테이블 생성
        html = ["<table>"]
        processed = set()
        
        for i, row in enumerate(grid):
            html.append("<tr>")
            for j, token in enumerate(row):
                if (i, j) in processed:
                    continue
                
                if token == 'C':
                    rowspan, colspan = spans[(i, j)]
                    cell_tag = "<td"
                    
                    if rowspan > 1:
                        cell_tag += f" rowspan='{rowspan}'"
                    if colspan > 1:
                        cell_tag += f" colspan='{colspan}'"
                    
                    cell_tag += ">"
                    
                    if (i, j) in text_cells:
                        text_idx = text_cells[(i, j)]
                        text = text_regions[text_idx].get('text', '').strip()
                        html.append(f"{cell_tag}{text}</td>")
                    else:
                        html.append(f"{cell_tag}</td>")
                    
                    for r in range(i, i + rowspan):
                        for c in range(j, j + colspan):
                            processed.add((r, c))
            
            html.append("</tr>")
        
        html.append("</table>")
        return '\n'.join(html)
        
    except Exception as e:
        print(f"Error constructing pred HTML: {str(e)}")
        return f"<table><tr><td>Error: {str(e)}</td></tr></table>"

def extract_spans_from_otsl(otsl_tokens: List[int], tokenizer: OTSLTokenizer) -> Tuple[np.ndarray, np.ndarray]:
    """OTSL 토큰에서 span matrix 추출"""
    # 1. OTSL 토큰을 그리드로 변환
    tokens = []
    for token_id in otsl_tokens:
        if token_id in tokenizer.id2token:
            token = tokenizer.id2token[token_id]
            if token not in ['[BOS]', '[EOS]', '[PAD]']:
                tokens.append(token)
    
    # 그리드 생성 
    grid = []
    current_row = []
    
    for token in tokens:
        if token == 'NL':
            if current_row:
                current_row.append('NL')  # NL 태그를 행의 마지막에 추가
                grid.append(current_row)
            current_row = []
        else:
            current_row.append(token)
            
    if current_row:  # 마지막 행 처리
        current_row.append('NL')
        grid.append(current_row)
    
    if not grid:
        return np.array([]), np.array([])
    
    # print("Grid")
    # print("=*10")
    # for g in grid:
    #     for c in g:
    #         print(c, end=" ")
    #     print()
    # 모든 행의 길이가 같은지 확인
    num_cols = len(grid[0])  # NL 포함
    num_rows = len(grid)
    
    # Row span matrix 생성
    row_span_matrix = np.ones((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] == 'NL':
                row_span_matrix[i][j] = 0  # NL은 span 0
            elif grid[i][j] == 'C':
                span_size = 1
                k = i + 1
                while k < num_rows and grid[k][j] in ['U', 'X']:
                    span_size += 1
                    k += 1
                for c in range(i, k):
                    row_span_matrix[c][j] = span_size

    # Column span matrix 생성 
    col_span_matrix = np.ones((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] == 'NL':
                col_span_matrix[i][j] = 0  # NL은 span 0
            elif grid[i][j] == 'C':
                span_size = 1
                k = j + 1
                while k < num_cols and grid[i][k] in ['L', 'X']:
                    span_size += 1
                    k += 1
                for r in range(j, k):
                    col_span_matrix[i][r] = span_size
                    
    return row_span_matrix, col_span_matrix



def compute_span_coefficients(
    row_span_matrix: np.ndarray,  # (rows, cols)
    col_span_matrix: np.ndarray,  # (rows, cols)
) -> Tuple[np.ndarray, np.ndarray]:
    """span coefficient matrices 계산
    
    Args:
        row_span_matrix: 각 셀의 row span 값을 나타내는 행렬 (NL 태그 포함)
        col_span_matrix: 각 셀의 column span 값을 나타내는 행렬 (NL 태그 포함)
        
    Returns:
        row_coef: (rows, cols, rows, cols) - row-wise coefficients
        col_coef: (rows, cols, rows, cols) - column-wise coefficients
    """
    rows, cols = row_span_matrix.shape
    real_cols = cols - 1  # NL 태그 제외
    
    # 결과 matrices 초기화
    row_coef = np.zeros((rows, real_cols, rows, real_cols))
    col_coef = np.zeros((rows, real_cols, rows, real_cols))
    
    # 각 셀 pair에 대해 coefficient 계산
    for i in range(rows):
        for j in range(real_cols):
            for p in range(rows):
                for q in range(real_cols):
                    # Row-wise coefficient 계산 (column 방향으로 투영)
                    if i == p:  # 같은 행의 셀들
                        # 1. 같은 span 내에 있는지 확인
                        if (col_span_matrix[i,j] == col_span_matrix[p,q] and 
                            row_span_matrix[i,j] == row_span_matrix[p,q]):
                            # 시작점이 같은지 확인 (왼쪽으로 가면서)
                            curr_j, curr_q = j, q
                            while curr_j > 0 and col_span_matrix[i, curr_j-1] > 1:
                                curr_j -= 1
                            while curr_q > 0 and col_span_matrix[p, curr_q-1] > 1:
                                curr_q -= 1
                            
                            if curr_j == curr_q:  # 같은 span 내에 있음
                                row_coef[i,j,p,q] = 1.0
                                continue
                        
                        # 2. column 방향으로 투영했을 때 overlap 계산
                        span_i = row_span_matrix[i,j]  # 현재 셀의 row span
                        span_p = row_span_matrix[p,q]  # 대상 셀의 row span
                        
                        # 두 셀이 column 방향으로 겹치는지 확인
                        if min(i + span_i, p + span_p) > max(i, p):
                            overlap = 1  # column-wise projection에서는 겹치면 1
                            row_coef[i,j,p,q] = overlap / (span_i * span_p)
                    
                    # Column-wise coefficient 계산 (row 방향으로 투영)
                    if j == q:  # 같은 열의 셀들
                        # 1. 같은 span 내에 있는지 확인
                        if (row_span_matrix[i,j] == row_span_matrix[p,q] and 
                            col_span_matrix[i,j] == col_span_matrix[p,q]):
                            # 시작점이 같은지 확인 (위로 올라가면서)
                            curr_i, curr_p = i, p
                            while curr_i > 0 and row_span_matrix[curr_i-1, j] > 1:
                                curr_i -= 1
                            while curr_p > 0 and row_span_matrix[curr_p-1, q] > 1:
                                curr_p -= 1
                            
                            if curr_i == curr_p:  # 같은 span 내에 있음
                                col_coef[i,j,p,q] = 1.0
                                continue
                        
                        # 2. row 방향으로 투영했을 때 overlap 계산
                        span_i = col_span_matrix[i,j]  # 현재 셀의 column span
                        span_p = col_span_matrix[p,q]  # 대상 셀의 column span
                        
                        # 두 셀이 row 방향으로 겹치는지 확인
                        if min(j + span_i, q + span_p) > max(j, q):
                            overlap = min(j + span_i, q + span_p) - max(j, q)
                            col_coef[i,j,p,q] = overlap / (span_i * span_p)
    
    return row_coef, col_coef

def pad_coef_matrix(
    coef: np.ndarray,  # (rows, cols, rows, cols)
    target_size: int,  # 패딩 후 목표 크기
) -> torch.Tensor:
    """coefficient matrix를 목표 크기로 패딩"""
    rows, cols, _, _ = coef.shape
    padded = torch.full((target_size, target_size), -1)
    
    # 4D를 2D로 변환하면서 패딩
    for i in range(rows):
        for j in range(cols):
            for p in range(rows):
                for q in range(cols):
                    if coef[i,j,p,q] > 0:
                        # (i,j)와 (p,q)의 위치를 1D 인덱스로 변환
                        idx1 = i * cols + j
                        idx2 = p * cols + q
                        padded[idx1, idx2] = coef[i,j,p,q]
    
    return padded


def get_coef_matrix(
    otsl_tokens: torch.Tensor,  # (B, 688)
    tokenizer: OTSLTokenizer,
    target_size: int,  # layout_prompt_length 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OTSL 토큰에서 span coefficient matrix 추출"""
    B = otsl_tokens.size(0)
    row_coefs, col_coefs = [], []
    
    for b in range(B):
        # 1. tensor -> list 변환 및 디코딩
        tokens = otsl_tokens[b].tolist()
        # 2. span matrix 계산
        row_span_matrix, col_span_matrix = extract_spans_from_otsl(tokens, tokenizer)
        # print("="*50)
        # print("row span matrix")
        # print(row_span_matrix)
        # print("col span matrix")
        # print(col_span_matrix)
        # 3. coefficient matrix 계산
        row_coef, col_coef = compute_span_coefficients(
            row_span_matrix=row_span_matrix,
            col_span_matrix=col_span_matrix
        )
        # print("row coef shape")
        # print(row_coef.shape)
        # print("col coef shape")
        # print(col_coef.shape)
        # 4. 패딩
        row_coef_padded = pad_coef_matrix(row_coef, target_size)
        col_coef_padded = pad_coef_matrix(col_coef, target_size)
        # print("row coef padded shape")
        # print(row_coef_padded.shape)
        # print("col coef padded shape")
        # print(col_coef_padded.shape)
        
        # print("\n=== Original Coefficient Matrices (before padding) ===")
        # print("Row-wise coefficients:")
        # for i in range(row_coef.shape[0]):
        #     for j in range(row_coef.shape[1]):
        #         if np.any(row_coef[i,j] > 0):  # 유효한 계수가 있는 경우만
        #             print(f"Cell ({i},{j}) coefficients:")
        #             for p in range(row_coef.shape[2]):
        #                 for q in range(row_coef.shape[3]):
        #                     if row_coef[i,j,p,q] > 0:
        #                         print(f"  -> ({p},{q}): {row_coef[i,j,p,q]:.2f}")
        
        # print("\nColumn-wise coefficients:")
        # for i in range(col_coef.shape[0]):
        #     for j in range(col_coef.shape[1]):
        #         if np.any(col_coef[i,j] > 0):  # 유효한 계수가 있는 경우만
        #             print(f"Cell ({i},{j}) coefficients:")
        #             for p in range(col_coef.shape[2]):
        #                 for q in range(col_coef.shape[3]):
        #                     if col_coef[i,j,p,q] > 0:
        #                         print(f"  -> ({p},{q}): {col_coef[i,j,p,q]:.2f}")
        # raise ValueError("debug")
        row_coefs.append(row_coef_padded)
        col_coefs.append(col_coef_padded)
    
    # 5. 배치 단위로 stack
    row_coefs = torch.stack(row_coefs)  # (B, 688, 688)
    col_coefs = torch.stack(col_coefs)  # (B, 688, 688)
    
    return row_coefs, col_coefs
