import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from models.otsl_tokenizer import OTSLTokenizer
from typing import Dict, Optional
import jsonlines
from utils.util import extract_spans_from_html, convert_html_to_otsl



DEBUG =  False # 실제 학습 시에는 False로 변경 필요
DEBUG_SAMPLE_SIZE = {
            'train': 100,  # 학습용 샘플 수
            'val': 10    # 검증용 샘플 수
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
        
    def _load_annotations(self):
        """주석 파일 로드 및 검증"""
        jsonl_file = os.path.join(self.data_dir, f'{self.split}.jsonl')
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
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터셋에서 하나의 샘플을 가져옴"""
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]
        
        # 1. 이미지 로드 및 전처리
        image = Image.open(os.path.join(self.data_dir, self.split, image_id + '.png')).convert('RGB')
        image_width, image_height = image.size
        image = self.transform(image)
        
        # 2. OTSL 토큰 생성
        otsl_sequence = convert_html_to_otsl(ann)
        tokens = self.tokenizer.encode(otsl_sequence)
        
        # 3. HTML에서 span 정보 추출
        processed_cells, row_span_matrix, col_span_matrix = extract_spans_from_html(ann['html']['structure'])
        
        # 4. annotation에서 cell 정보 추출 및 정규화
        cells = []
        bboxes = []
        for cell in ann['cells']:  # 'cells' 리스트에서 직접 접근
            if 'bbox' in cell:  # non-empty cells
                x0, y0, x1, y1 = cell['bbox']
                normalized_bbox = [
                    x0 / image_width,
                    y0 / image_height,
                    x1 / image_width,
                    y1 / image_height
                ]
                bboxes.append(normalized_bbox)
                cells.append({
                    'text': cell['text'],
                    'bbox': normalized_bbox
                })
        
        return {
            'image_id': image_id,
            'image': image,
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'row_spans': torch.tensor(row_span_matrix, dtype=torch.float32),
            'col_spans': torch.tensor(col_span_matrix, dtype=torch.float32),
            'cells': cells,  # cell 정보 추가 (text와 bbox 포함)
            'html': ann['html'],  # 디버깅용
        }

    def __len__(self) -> int:
        return len(self.image_ids)
    