import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from models.otsl_tokenizer import OTSLTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
import jsonlines


DEBUG = True  # 실제 학습 시에는 False로 변경 필요
DEBUG_SAMPLE_SIZE = {
            'train': 1000,  # 학습용 샘플 수
            'val': 50     # 검증용 샘플 수
        }  # 디버깅용 샘플 수
class TableDataset(Dataset):
    """
    테이블 이미지 데이터셋
    
    논문 Section 4.1 Datasets 참조:
    - PubTabNet, FinTabNet, SynthTabNet 데이터셋 지원
    - cell-level annotations 또는 OCR engine으로부터 text regions 획득
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_seq_length: int = 1376,  # 논문 4.2
        image_size: int = 768,       # 논문 4.2
        use_ocr: bool = False,       # OCR engine 사용 여부
        tokenizer: Optional[OTSLTokenizer] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.use_ocr = use_ocr
        
        # Tokenizer 초기화
        self.tokenizer = tokenizer or OTSLTokenizer()
        
        # 이미지 전처리 (논문 4.2)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 주석 파일 로드
        self._load_annotations()
        
    # def _load_annotations(self):
    #     """주석 파일 로드 및 검증"""
    #     ann_file = os.path.join(self.data_dir, 'annotations',f'{self.split}.json')
    #     with open(ann_file, 'r', encoding='utf-8') as f:
    #         self.annotations = json.load(f)
            
        # self.image_ids = list(self.annotations.keys())
    
    # PubTabNet 2.0.0 데이터셋 로드
    def _load_annotations(self):
        """주석 파일 로드 및 검증"""
        jsonl_file = os.path.join(self.data_dir, 'PubTabNet_2.0.0.jsonl')
        self.annotations = {}
        self.image_ids = []
        
        # DEBUG MODE: 빠른 테스트를 위한 설정
        if DEBUG: os.makedirs('./debugs', exist_ok=True)
        
        
        print(f"\nLoading {self.split} dataset...")
        valid_samples = 0  # 이미지가 실제로 존재하는 샘플 수
        valid_images_path = []
        with jsonlines.open(jsonl_file) as reader:
            for annotation in reader:
                # 현재 split에 해당하는 데이터만 필터링
                if annotation['split'] == self.split:
                    image_id = annotation['filename'].split('.')[0]  # .png 확장자 제거
                    image_path = os.path.join(self.data_dir, self.split, image_id + '.png')
                    
                    # 이미지 파일이 존재하는지 확인               
                    if not os.path.exists(image_path):
                        continue
                    
                    # DEBUG MODE: 지정된 유효한 샘플 수에 도달하면 중단
                    if DEBUG and valid_samples >= DEBUG_SAMPLE_SIZE.get(self.split, 1):
                        break
                    
                    # HTML 구조에서 cell 정보 추출
                    cells = []
                    for cell_data in annotation['html']['cells']:
                        cell_info = {
                            'text': ' '.join(cell_data.get('tokens', [])),
                            'bbox': cell_data.get('bbox', [0, 0, 0, 0]),
                        }
                        cells.append(cell_info)
                    
                    # 필요한 정보만 저장
                    processed_ann = {
                        'html': annotation['html'],  # 원본 HTML 구조 보존
                        'cells': cells
                    }
                    
                    self.annotations[image_id] = processed_ann
                    self.image_ids.append(image_id)
                    valid_samples += 1
                    
        print(f"{self.split} 데이터셋 로드 완료:")
        print(f"- 유효한 샘플 수: {valid_samples}")

    def _extract_spans_from_bbox(self, cells: List[Dict]) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """바운딩 박스로부터 행/열 span 정보 추출
        
        Args:
            cells: HTML 구조의 cell 정보 리스트
            
        Returns:
            processed_cells: row_idx, col_idx가 추가된 cell 정보
            row_span_matrix: 행 span 정보 행렬
            col_span_matrix: 열 span 정보 행렬
        """
        # 1. 빈 셀 제외하고 bbox 기준으로 정렬
        valid_cells = [cell for cell in cells if 'bbox' in cell]
        valid_cells = sorted(valid_cells, key=lambda x: (x['bbox'][1], x['bbox'][0]))  # y0, x0 기준 정렬
        
        if not valid_cells:
            return [], np.array([]), np.array([])
        
        # 2. 행 그룹화 (y 좌표가 비슷한 셀들을 같은 행으로 그룹화)
        rows = []
        current_row = []
        current_y = valid_cells[0]['bbox'][1]
        
        for cell in valid_cells:
            if abs(cell['bbox'][1] - current_y) > 5:  # 새로운 행 시작 (threshold=5)
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))  # x0 기준 정렬
                current_row = [cell]
                current_y = cell['bbox'][1]
            else:
                current_row.append(cell)
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['bbox'][0]))
        
        # 3. row_idx, col_idx 할당
        processed_cells = []
        max_rows = len(rows)
        max_cols = max(len(row) for row in rows)
        
        # 그리드 초기화 (빈 셀 포함)
        grid = [[None] * max_cols for _ in range(max_rows)]
        
        # 셀 정보 채우기
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                cell_info = {
                    'bbox': cell['bbox'],
                    'text': ' '.join(cell.get('tokens', [])),
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'rowspan': 1,  # 기본값
                    'colspan': 1   # 기본값
                }
                grid[row_idx][col_idx] = cell_info
                processed_cells.append(cell_info)
        
        # 4. Span 정보 계산
        # 수평 span 감지
        for row_idx in range(max_rows):
            col_idx = 0
            while col_idx < max_cols:
                if grid[row_idx][col_idx] is not None:
                    current_cell = grid[row_idx][col_idx]
                    colspan = 1
                    # 다음 열이 비어있거나 x 좌표가 매우 가까운 경우 병합
                    while (col_idx + colspan < max_cols and 
                           (grid[row_idx][col_idx + colspan] is None or 
                            abs(current_cell['bbox'][2] - grid[row_idx][col_idx + colspan]['bbox'][0]) < 5)):
                        colspan += 1
                    current_cell['colspan'] = colspan
                    col_idx += colspan
                else:
                    col_idx += 1
        
        # 수직 span 감지
        for col_idx in range(max_cols):
            row_idx = 0
            while row_idx < max_rows:
                if grid[row_idx][col_idx] is not None:
                    current_cell = grid[row_idx][col_idx]
                    rowspan = 1
                    # 다음 행이 비어있거나 y 좌표가 매우 가까운 경우 병합
                    while (row_idx + rowspan < max_rows and 
                           (grid[row_idx + rowspan][col_idx] is None or 
                            abs(current_cell['bbox'][3] - grid[row_idx + rowspan][col_idx]['bbox'][1]) < 5)):
                        rowspan += 1
                    current_cell['rowspan'] = rowspan
                    row_idx += rowspan
                else:
                    row_idx += 1
        
        # 5. Span matrices 생성
        row_span_matrix = np.zeros((max_rows, max_cols))
        col_span_matrix = np.zeros((max_rows, max_cols))
        
        for cell in processed_cells:
            r, c = cell['row_idx'], cell['col_idx']
            rs, cs = cell['rowspan'], cell['colspan']
            
            if rs > 1:
                row_span_matrix[r:r+rs, c] = 1
            if cs > 1:
                col_span_matrix[r, c:c+cs] = 1
        
        return processed_cells, row_span_matrix, col_span_matrix

    def _convert_html_to_otsl(self, ann: Dict, image_id: str = None) -> str:
        """HTML 구조를 OTSL 토큰으로 변환
        
        Args:
            ann: HTML 구조 정보를 담은 딕셔너리
            image_id: 디버깅용 이미지 ID
            
        Returns:
            str: OTSL 토큰 시퀀스
        """
        # 1. 셀 정보 추출
        cells = []
        for cell in ann['html']['cells']:
            cell_info = {
                'bbox': cell.get('bbox', [0, 0, 0, 0]),
                'tokens': cell.get('tokens', [])
            }
            cells.append(cell_info)
        
        # 2. span 정보 추출
        processed_cells, row_spans, col_spans = self._extract_spans_from_bbox(cells)
        
        if not processed_cells:
            return ""
        
        # 3. OTSL 그리드 초기화
        max_rows = max(cell['row_idx'] for cell in processed_cells) + 1
        max_cols = max(cell['col_idx'] for cell in processed_cells) + 1
        grid = [[None] * max_cols for _ in range(max_rows)]
        
        # 4. 그리드에 OTSL 토큰 할당
        for cell in processed_cells:
            r, c = cell['row_idx'], cell['col_idx']
            rs, cs = cell['rowspan'], cell['colspan']
            
            # 기본 셀 ('C')
            if rs == 1 and cs == 1:
                grid[r][c] = 'C'
                continue
            
            # 수평 병합 ('L')
            if rs == 1 and cs > 1:
                grid[r][c] = 'C'
                for j in range(c + 1, c + cs):
                    grid[r][j] = 'L'
                continue
            
            # 수직 병합 ('U')
            if rs > 1 and cs == 1:
                grid[r][c] = 'C'
                for i in range(r + 1, r + rs):
                    grid[i][c] = 'U'
                continue
            
            # 2D 병합 ('X')
            if rs > 1 and cs > 1:
                grid[r][c] = 'C'
                # 첫 행의 나머지 셀들
                for j in range(c + 1, c + cs):
                    grid[r][j] = 'L'
                # 나머지 행들
                for i in range(r + 1, r + rs):
                    grid[i][c] = 'U'
                    for j in range(c + 1, c + cs):
                        grid[i][j] = 'X'
        
        # DEBUG MODE: OTSL 변환 결과를 이미지로 시각화
        if image_id is not None:
            # 원본 이미지 로드
            img_path = os.path.join(self.data_dir, self.split, image_id + '.png')
            img = Image.open(img_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # 폰트 설정 (기본 폰트 사용)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 각 셀에 대해 박스와 OTSL 태그 그리기
            for cell in processed_cells:
                bbox = cell['bbox']
                r, c = cell['row_idx'], cell['col_idx']
                tag = grid[r][c]
                
                # 박스 그리기
                draw.rectangle(bbox, outline='red', width=2)
                
                # OTSL 태그 텍스트 그리기
                text_pos = (bbox[0], bbox[1] - 20)  # 박스 위에 태그 표시
                draw.text(text_pos, tag, fill='blue', font=font)
            
            # 디버그 이미지 저장
            os.makedirs('./debugs/otsl', exist_ok=True)
            debug_img_path = os.path.join(f"./debugs/otsl/{image_id}.png")
            img.save(debug_img_path)
            # print(f"디버그 이미지 저장됨: {debug_img_path}")
        
        # 5. OTSL 시퀀스 생성
        tokens = []
        for row in grid:
            for token in row:
                tokens.append(token or 'C')  # None을 'C'로 대체
            tokens.append('NL')  # 행 구분자
        
        return ' '.join(tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터셋에서 하나의 샘플을 가져옴"""
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]
        
        # 1. 이미지 로드 및 전처리
        image = Image.open(os.path.join(self.data_dir, self.split, image_id + '.png')).convert('RGB')
        image_width, image_height = image.size
        image = self.transform(image)
        
        # 2. OTSL 토큰 생성 (디버그 모드에서는 image_id 전달)
        
        otsl_sequence = self._convert_html_to_otsl(ann, None)
        tokens = self.tokenizer.encode(otsl_sequence)
        
        # 3. 바운딩 박스 정규화
        processed_cells, row_span_matrix, col_span_matrix = self._extract_spans_from_bbox(ann['cells'])
        bboxes = []
        for cell in processed_cells:
            x0, y0, x1, y1 = cell['bbox']
            normalized_bbox = [
                x0 / image_width,
                y0 / image_height,
                x1 / image_width,
                y1 / image_height
            ]
            bboxes.append(normalized_bbox)
        
        return {
            'image': image,
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'row_spans': torch.tensor(row_span_matrix, dtype=torch.float32),
            'col_spans': torch.tensor(col_span_matrix, dtype=torch.float32),
            'html': ann['html'],  # 디버깅용
            'cells': processed_cells  # 디버깅용
        }

    def __len__(self) -> int:
        return len(self.image_ids)