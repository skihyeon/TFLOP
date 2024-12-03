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

def convert_html_to_otsl(ann: Dict) -> Tuple[str, List[bool]]:
    """HTML 구조를 OTSL로 변환하고 데이터 존재 여부를 반환
    
    Args:
        ann: PubTabNet annotation 딕셔너리
        
    Returns:
        otsl_sequence: OTSL 토큰 시퀀스
        has_data_1D_list: 각 셀의 데이터 존재 여부 (1D 리스트)
    """
    try:
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
                        del active_rowspans[col]
                
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
        
        # 각 행과 열을 순회하면서 OTSL 토큰 결정
        for row in range(num_rows):
            for col in range(num_cols):
                if grid[row][col] is not None:
                    continue
                    
                # 위쪽 셀의 rowspan 확인
                is_under_rowspan = False
                if row > 0:
                    for r in range(row-1, -1, -1):
                        if (r, col) in spans:
                            rowspan, _, _ = spans[(r, col)]
                            if row < r + rowspan:
                                is_under_rowspan = True
                                grid[row][col] = 'U'
                                break
                
                if is_under_rowspan:
                    continue
                
                # 현재 위치에 새로운 셀이 있는 경우
                if (row, col) in spans:
                    grid[row][col] = 'C'
                    rowspan, colspan, has_content = spans[(row, col)]
                    has_data[row][col] = has_content
                    
                    # colspan 처리
                    for c in range(col + 1, col + colspan):
                        grid[row][c] = 'L'
                
                # 빈 셀 처리
                elif grid[row][col] is None:
                    grid[row][col] = 'C'
                    has_data[row][col] = False
        
        # 3. OTSL 시퀀스 생성
        otsl_tokens = []
        for row in grid:
            otsl_tokens.extend(token for token in row if token is not None)
            otsl_tokens.append('NL')
        
        has_data_1D_list = []
        for row in has_data:
            row.append(False)  # NL 토큰에 대한 has_data
            has_data_1D_list.extend(row)
        
        return " ".join(otsl_tokens), has_data_1D_list
        
    except Exception as e:
        print(f"Error converting HTML to OTSL: {str(e)}")
        return None

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
        tokens = [t for t in otsl_sequence.split() if t not in ['[BOS]', '[EOS]', '[PAD]']]
        grid = []
        current_row = []
        
        # 2. 토큰 인덱스와 그리드 위치 매핑 (모든 태그 포함)
        token_positions = {}  # token_idx -> (row, col) 매핑
        token_idx = 0
        current_row_idx = 0
        
        for token in tokens:
            if token == 'NL':
                if current_row:
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
    try:
        # 1. OTSL 토큰을 그리드로 변환
        tokens = []
        for token_id in otsl_tokens:
            if token_id in tokenizer.id2token:
                token = tokenizer.id2token[token_id]
                if token not in ['[BOS]', '[EOS]', '[PAD]']:
                    tokens.append(token)
        
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
        
        if not grid:
            return np.array([]), np.array([])
        
        num_rows = len(grid)
        num_cols = len(grid[0])
        
        # Row span matrix 생성
        row_span_matrix = np.ones((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                if grid[i][j] == 'C':
                    try:
                        span_size = 1
                        k = j + 1
                        while k < num_cols and grid[i][k] in ['L', 'X']:
                            span_size += 1
                            k += 1
                        for c in range(j, k):
                            row_span_matrix[i][c] = span_size
                    except Exception as e:
                        print(f"Error in row span calculation at ({i}, {j}):")
                        print(f"Grid shape: {num_rows}x{num_cols}")
                        print(f"Current row: {grid[i]}")
                        print(f"Error: {str(e)}")
                        raise
        
        # Column span matrix 생성
        col_span_matrix = np.ones((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                if grid[i][j] == 'C':
                    try:
                        span_size = 1
                        k = i + 1
                        while k < num_rows and grid[k][j] in ['U', 'X']:
                            span_size += 1
                            k += 1
                        for r in range(i, k):
                            col_span_matrix[r][j] = span_size
                    except Exception as e:
                        print(f"Error in column span calculation at ({i}, {j}):")
                        print(f"Grid shape: {num_rows}x{num_cols}")
                        print(f"Current column: {[grid[r][j] for r in range(num_rows)]}")
                        print(f"Error: {str(e)}")
                        raise
                        
        return row_span_matrix, col_span_matrix
        
    except Exception as e:
        print("\n=== Error in extract_spans_from_otsl ===")
        print(f"Input tokens: {otsl_tokens}")
        print(f"Decoded tokens: {tokens}")
        print("\nGrid:")
        for row in grid:
            print(row)
        print(f"\nError: {str(e)}")
        raise
    
# def extract_spans_from_otsl(otsl_tokens: List[int], tokenizer: OTSLTokenizer) -> Tuple[np.ndarray, np.ndarray]:
#     """OTSL 토큰에서 span matrix 추출"""
#     # 1. OTSL 토큰을 그리드로 변환
#     tokens = []
#     for token_id in otsl_tokens:
#         if token_id in tokenizer.id2token:
#             token = tokenizer.id2token[token_id]
#             if token not in ['[BOS]', '[EOS]', '[PAD]']:
#                 tokens.append(token)
    
#     grid = []
#     current_row = []
#     for token in tokens:
#         if token == 'NL':
#             if current_row:
#                 grid.append(current_row)
#                 current_row = []
#         else:
#             current_row.append(token)
#     if current_row:
#         grid.append(current_row)
    
#     if not grid:
#         return np.array([]), np.array([])
    
#     num_rows = len(grid)
#     num_cols = len(grid[0])
    
#     # Row span matrix 생성
#     row_span_matrix = np.ones((num_rows, num_cols))
#     for i in range(num_rows):
#         for j in range(num_cols):
#             if grid[i][j] == 'C':
#                 # 현재 셀부터 시작하여 연속된 span 크기 계산
#                 span_size = 1
#                 # 오른쪽으로 L 토큰 카운트
#                 k = j + 1
#                 while k < num_cols and grid[i][k] in ['L', 'X']:
#                     span_size += 1
#                     k += 1
#                 # 시작 셀(C)과 모든 span된 셀(L)에 동일한 크기 할당
#                 for c in range(j, k):
#                     row_span_matrix[i][c] = span_size
    
#     # Column span matrix 생성
#     col_span_matrix = np.ones((num_rows, num_cols))
#     for i in range(num_rows):
#         for j in range(num_cols):
#             if grid[i][j] == 'C':
#                 # 현재 셀부터 시작하여 연속된 span 크기 계산
#                 span_size = 1
#                 # 아래로 U 토큰 카운트
#                 k = i + 1
#                 while k < num_rows and grid[k][j] in ['U', 'X']:
#                     span_size += 1
#                     k += 1
#                 # 시작 셀(C)과 모든 span된 셀(U)에 동일한 크기 할당
#                 for r in range(i, k):
#                     col_span_matrix[r][j] = span_size
    
#     # print("Grid:")
#     # for row in grid:
#     #     print(row)
#     # print("Row span matrix:")
#     # for row in row_span_matrix:
#     #     print(row)
#     # print("Column span matrix:")
#     # for row in col_span_matrix:
#     #     print(row)
#     return row_span_matrix, col_span_matrix

def compute_span_coefficients(
    row_span_matrix: np.ndarray,  # (num_rows, num_cols)
    col_span_matrix: np.ndarray,  # (num_rows, num_cols)
    box_indices: torch.Tensor,    # (N, M)
    num_boxes: int                # N
) -> Tuple[torch.Tensor, torch.Tensor]:
    """실제 bbox 간의 coefficient matrix 계산 (many-to-one 관계 지원)"""
    num_rows, num_cols = row_span_matrix.shape
    device = box_indices.device  # box_indices의 device 사용
    
    # box_indices를 flatten하고 유효한 인덱스만 선택
    box_indices_flat = box_indices.view(-1)  # (N*M,)
    valid_mask = box_indices_flat != -1
    valid_indices = box_indices_flat[valid_mask]
    
    if len(valid_indices) == 0:
        return torch.zeros((num_boxes, num_boxes), device=device), torch.zeros((num_boxes, num_boxes), device=device)
    
    # 1D -> 2D 변환을 벡터화
    rows = (valid_indices // num_cols).cpu().numpy()
    cols = (valid_indices % num_cols).cpu().numpy()
    
    # 범위를 벗어나는 인덱스 필터링
    valid_range = (rows < num_rows) & (cols < num_cols)
    rows = rows[valid_range]
    cols = cols[valid_range]
    
    # Span 정보 추출 (float32로 변환)
    row_spans = row_span_matrix[rows, cols].astype(np.float32)  # (V,)
    col_spans = col_span_matrix[rows, cols].astype(np.float32)  # (V,)
    
    # 행/열 매칭 행렬 생성 (V, V)
    same_row = (rows[:, None] == rows[None, :])  # (V, V)
    same_col = (cols[:, None] == cols[None, :])  # (V, V)
    
    # Span overlap 계산 (float32로 유지)
    row_overlap = np.minimum(row_spans[:, None], row_spans[None, :])  # (V, V)
    col_overlap = np.minimum(col_spans[:, None], col_spans[None, :])  # (V, V)
    
    # Coefficient 계산 (float32로 유지)
    row_coef_valid = np.where(
        same_row,
        row_overlap / (row_spans[:, None] * row_spans[None, :]),
        0
    ).astype(np.float32)  # (V, V)
    
    col_coef_valid = np.where(
        same_col,
        col_overlap / (col_spans[:, None] * col_spans[None, :]),
        0
    ).astype(np.float32)  # (V, V)
    
    # 최종 크기의 coefficient 행렬 생성
    row_coef = torch.zeros((num_boxes, num_boxes), dtype=torch.float32, device=device)
    col_coef = torch.zeros((num_boxes, num_boxes), dtype=torch.float32, device=device)
    
    # box_indices의 원래 box 인덱스 복원 (device 일치시킴)
    box_idx_map = torch.div(
        torch.arange(len(box_indices_flat), device=device),
        box_indices.size(1),
        rounding_mode='floor'
    )[valid_mask][valid_range]
    
    # 유효한 인덱스에 대해서만 값 할당 (many-to-one 관계 고려)
    for i, (src_idx, tgt_idx) in enumerate(zip(box_idx_map.cpu(), box_idx_map.cpu())):
        for j, (src_idx2, tgt_idx2) in enumerate(zip(box_idx_map.cpu(), box_idx_map.cpu())):
            row_coef[src_idx, src_idx2] = max(row_coef[src_idx, src_idx2], torch.tensor(row_coef_valid[i, j], device=device))
            col_coef[src_idx, src_idx2] = max(col_coef[src_idx, src_idx2], torch.tensor(col_coef_valid[i, j], device=device))
    
    return row_coef, col_coef

def get_coef_matrix(
    otsl_tokens: torch.Tensor,  # (B, L)
    tokenizer: OTSLTokenizer,
    box_indices: torch.Tensor,  # (B, N, M)
    num_boxes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OTSL 토큰에서 span coefficient matrix 추출 (many-to-one 관계 지원)"""
    B = otsl_tokens.size(0)
    row_coefs, col_coefs = [], []
    
    for b in range(B):
        # 1. tensor -> list 변환 및 디코딩
        tokens = otsl_tokens[b].tolist()
        
        # 2. span matrix 계산
        row_span_matrix, col_span_matrix = extract_spans_from_otsl(tokens, tokenizer)
        
        # 3. coefficient matrix 계산 (현재 배치의 box_indices 사용)
        row_coef, col_coef = compute_span_coefficients(
            row_span_matrix=row_span_matrix,
            col_span_matrix=col_span_matrix,
            box_indices=box_indices[b],  # (N, M)
            num_boxes=num_boxes
        )
        
        row_coefs.append(row_coef)
        col_coefs.append(col_coef)
    
    # 4. 배치 단위로 stack
    row_coefs = torch.stack(row_coefs)  # (B, N, N)
    col_coefs = torch.stack(col_coefs)  # (B, N, N)
    
    return row_coefs, col_coefs