import os
import wandb
from datetime import datetime
from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import torch
from models.otsl_tokenizer import OTSLTokenizer

def init_wandb(model_config, train_config):
    """wandb 초기화"""
    try:
        # 기존 wandb 프로세스 정리
        if wandb.run is not None:
            wandb.finish()
            
        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(
            entity=os.environ.get('WANDB_ENTITY'),
            project="TFLOP",
            name=datetime.now().strftime('%Y%m%d_%H%M%S'),
            config={
                "model_config": model_config.__dict__,
                "train_config": train_config.__dict__
            },
            settings=wandb.Settings(start_method="fork")  # 프로세스 관리 방식 변경
        )
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None


def convert_html_to_otsl(ann: Dict) -> Tuple[List[str], List[bool]]:
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
        
        def clean_text(tokens: List[str]) -> str:
            text = ''.join(tokens)
            for tag in ['<b>', '</b>', '<i>', '</i>', '<sup>', '</sup>', '<sub>', '</sub>']:
                text = text.replace(tag, '')
            return text.strip()
        
        structure_tokens = html_data['structure']['tokens']
        cells = html_data['cells']
        i = 0
        
        while i < len(structure_tokens):
            token = structure_tokens[i]
            
            if token in ['<thead>', '</thead>', '<tbody>', '</tbody>']:
                i += 1
                continue
                
            if token == '<tr>':
                html.append("<tr>")
                i += 1
                continue
                
            if token == '</tr>':
                html.append("</tr>")
                i += 1
                continue
            
            if token == '<td' or token == '<td>':
                cell_tag = "<td"
                
                # 현재 토큰이 '<td'인 경우 속성 확인
                if token == '<td':
                    i += 1
                    while i < len(structure_tokens) and structure_tokens[i] != '>':
                        attr = structure_tokens[i]
                        if 'rowspan="' in attr:
                            rowspan = attr.split('rowspan="')[1].split('"')[0]
                            cell_tag += f' rowspan="{rowspan}"'
                        elif 'colspan="' in attr:
                            colspan = attr.split('colspan="')[1].split('"')[0]
                            cell_tag += f' colspan="{colspan}"'
                        i += 1
                
                cell_tag += ">"
                
                # 셀 내용 추가
                if current_cell_idx < len(cells):
                    cell_content = clean_text(cells[current_cell_idx]['tokens'])
                    html.append(f"{cell_tag}{cell_content}</td>")
                    current_cell_idx += 1
                else:
                    html.append(f"{cell_tag}</td>")
            
            i += 1
        
        html.append("</table>")
        return '\n'.join(html)
        
    except Exception as e:
        print(f"Error constructing GT HTML: {str(e)}")
        return f"<table><tr><td>Error: {str(e)}</td></tr></table>"

def construct_table_html_pred(
    otsl_sequence: str,
    bbox_with_text: List[Dict[str, Union[str, List[float]]]],
    pointer_logits: torch.Tensor,
    empty_pointer_logits: torch.Tensor = None,
    confidence_threshold: float = 0.5
) -> str:
    """OTSL과 pointer logits를 사용하여 예측 테이블 생성"""
    # 1. OTSL 토큰을 그리드로 변환
    tokens = [t for t in otsl_sequence.split() if t not in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']]
    grid = []
    current_row = []
    
    # 2. 토큰 인덱스와 그리드 위치 매핑
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

    # 6. pointer_logits와 empty_pointer_logits를 사용하여 text_cells 매핑 생성
    text_cells = {}  # (row, col) -> List[(box_idx, text, prob)]
    empty_cells = set()  # empty로 판단된 셀들의 (row, col) 집합
    
    pointer_probs = torch.softmax(pointer_logits, dim=1)
    empty_probs = torch.sigmoid(empty_pointer_logits) if empty_pointer_logits is not None else None

    # 먼저 empty cell 판단
    if empty_probs is not None:
        for token_idx, empty_prob in enumerate(empty_probs):
            max_empty_prob = empty_prob.max()
            if max_empty_prob.item() >= confidence_threshold:
                if token_idx in token_positions:
                    row, col = token_positions[token_idx]
                    if row != -1:  # NL 태그가 아닌 경우
                        empty_cells.add((row, col))

    # 그 다음 text pointer 처리 (empty가 아닌 셀에 대해서만)
    for box_idx in range(pointer_probs.size(0)):
        max_prob, cell_idx = pointer_probs[box_idx].max(dim=0)
        if max_prob.item() >= confidence_threshold and cell_idx < len(token_positions):
            token_idx = cell_idx.item()
            if token_idx in token_positions:
                row, col = token_positions[token_idx]
                if row != -1 and (row, col) not in empty_cells:  # empty cell이 아닌 경우에만
                    if box_idx in bbox_with_text:
                        text = bbox_with_text[box_idx]['text']
                        if (row, col) not in text_cells:
                            text_cells[(row, col)] = []
                        text_cells[(row, col)].append((box_idx, text, max_prob.item()))

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
                
                if (i, j) in empty_cells:
                    html.append(f"{cell_tag}(empty)</td>")
                elif (i, j) in text_cells:
                    boxes = text_cells[(i, j)]
                    boxes.sort(key=lambda x: x[2], reverse=True)
                    combined_text = ' '.join(text for _, text, _ in boxes)
                    html.append(f"{cell_tag}{combined_text}</td>")
                else:
                    html.append(f"{cell_tag}</td>")
                
                for r in range(i, i + rowspan):
                    for c in range(j, j + colspan):
                        processed.add((r, c))
        
        html.append("</tr>")
    
    html.append("</table>")
    return '\n'.join(html)

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

def compute_span_coefficients(row_span_matrix: np.ndarray, col_span_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = row_span_matrix.shape
    real_cols = cols - 1  # NL 태그 제외
    
    # 1. Create coordinate matrices
    i_coords = np.arange(rows)[:, None, None, None]  # (rows, 1, 1, 1)
    j_coords = np.arange(real_cols)[None, :, None, None]  # (1, real_cols, 1, 1)
    p_coords = np.arange(rows)[None, None, :, None]  # (1, 1, rows, 1)
    q_coords = np.arange(real_cols)[None, None, None, :]  # (1, 1, 1, real_cols)
    
    # 2. Broadcast span values
    span_i_row = row_span_matrix[:, :real_cols, None, None]  # (rows, real_cols, 1, 1)
    span_p_row = row_span_matrix[None, None, :, :real_cols]  # (1, 1, rows, real_cols)
    
    span_i_col = col_span_matrix[:, :real_cols, None, None]  # (rows, real_cols, 1, 1)
    span_p_col = col_span_matrix[None, None, :, :real_cols]  # (1, 1, rows, real_cols)
    
    # 3. Row-wise overlap computation
    row_min = np.minimum(i_coords + span_i_row, p_coords + span_p_row)
    row_max = np.maximum(i_coords, p_coords)
    row_overlap = np.maximum(row_min - row_max, 0)
    row_coef = np.where(
        row_overlap > 0,
        row_overlap / (span_i_row * span_p_row),
        -1.0
    )
    
    # 4. Column-wise overlap computation
    col_min = np.minimum(j_coords + span_i_col, q_coords + span_p_col)
    col_max = np.maximum(j_coords, q_coords)
    col_overlap = np.maximum(col_min - col_max, 0)
    col_coef = np.where(
        col_overlap > 0,
        col_overlap / (span_i_col * span_p_col),
        -1.0
    )
    
    return row_coef, col_coef


def pad_coef_matrix(coef: np.ndarray, target_size: int) -> torch.Tensor:
    rows, cols, _, _ = coef.shape
    padded = torch.zeros((target_size, target_size), dtype=torch.float32)
    
    # 1. Reshape coef matrix to 2D
    coef_2d = coef.reshape(rows * cols, rows * cols)
    
    # 2. Create valid mask and get indices
    valid_mask = coef_2d >= 0
    valid_indices = np.nonzero(valid_mask)
    
    # 3. Convert to torch tensor with explicit dtype and copy values
    padded[valid_indices[0], valid_indices[1]] = torch.from_numpy(
        coef_2d[valid_mask].astype(np.float32)  # numpy dtype을 float32로 변환
    )
    
    return padded


def get_coef_matrix(
    otsl_tokens: torch.Tensor,  
    tokenizer: OTSLTokenizer,
    target_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = otsl_tokens.size(0)
    row_coefs, col_coefs = [], []

    for b in range(B):
        tokens = otsl_tokens[b].tolist()
        row_span_matrix, col_span_matrix = extract_spans_from_otsl(tokens, tokenizer)
        
        row_coefs.append(
            pad_coef_matrix(
                compute_span_coefficients(row_span_matrix, col_span_matrix)[0],
                target_size
            )
        )
        col_coefs.append(
            pad_coef_matrix(
                compute_span_coefficients(row_span_matrix, col_span_matrix)[1],
                target_size
            )
        )
    
    return torch.stack(row_coefs), torch.stack(col_coefs)