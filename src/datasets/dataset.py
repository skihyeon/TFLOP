import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from models.otsl_tokenizer import OTSLTokenizer
from typing import Dict, Optional, Any
import jsonlines
from utils.util import extract_spans_from_html, convert_html_to_otsl, extract_spans_from_otsl, compute_span_coefficients

# 디버그 모드 설정
DEBUG = True
DEBUG_SAMPLES = {
    'train': 10000,
    'val': 500
}

class TableDataset(Dataset):
    """
    테이블 이미지 데이터셋
    
    논문 Section 4.1 Datasets 참조:
    - PubTabNet, FinTabNet, SynthTabNet 데이터셋 지원
    - cell-level annotations 또는 OCR engine으로부터 text regions 획득
    - layout prompt를 위한 sequence length 및 position 조정 포함
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        total_sequence_length: int = 1376,  # 논문 4.2
        image_size: int = 768,       # 논문 4.2
        use_ocr: bool = False,       # OCR engine 사용 여부
        max_boxes: int = None  # 추가
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.total_sequence_length = total_sequence_length
        self.max_boxes = max_boxes  # 외부에서 주입 가능
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 주석 파일 로드 및 필터링
        self._load_and_filter_annotations()
        
    def _load_and_filter_annotations(self):
        """주석 파일 로드 및 sequence length 기준으로 필터링"""
        ann_file = os.path.join(self.data_dir, f'{self.split}_filtered.jsonl')
        
        # max_boxes가 지정되지 않은 경우에만 계산
        if self.max_boxes is None:
            # 첫 번째 패스: max_boxes 계산
            max_boxes = 0
            count = 0
            with jsonlines.open(ann_file) as reader:
                
                for ann in reader:
                    if 'html' in ann and 'cells' in ann['html']:
                        max_boxes = max(max_boxes, len(ann['html']['cells']))
                    count += 1
                    if DEBUG and count >= DEBUG_SAMPLES[self.split]:
                        break
            self.max_boxes = max_boxes
            # print(f"Calculated max_boxes: {self.max_boxes}")
        
        # 두 번째 패스: 실제 데이터 로딩
        self.annotations = {}
        self.image_names = []
        filtered_count = 0
        
        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                if DEBUG and len(self.image_names) >= DEBUG_SAMPLES[self.split]:
                    break
                    
                try:
                    otsl, has_data_flags = convert_html_to_otsl(ann)
                    if otsl is None:
                        filtered_count += 1
                        continue
                    
                    # 고정된 token 길이 계산
                    token_length = self.total_sequence_length - self.max_boxes
                    
                    # 기본 정보만 저장
                    image_name = ann['filename']
                    self.annotations[image_name] = {
                        'html': ann['html'],
                        'otsl': otsl,
                        'has_data_flags': has_data_flags
                    }
                    self.image_names.append(image_name)
                    
                except Exception:
                    filtered_count += 1
                    continue
        
        # print(f"Loaded {len(self.image_names)} samples for {self.split}")
        # print(f"Filtered out {filtered_count} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_names[idx]
        ann = self.annotations[image_name]
        
        # 이미지 로드 및 전처리
        image_path = os.path.join(self.data_dir, self.split, image_name)
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
        
        image_width, image_height = image.size
        image = self.transform(image)
        
        # 1. C 태그의 sequence 내 실제 위치 추적
        token_positions = []  # C 태그의 sequence 내 위치 저장
        for j, token_id in enumerate(ann['otsl']):
            if token_id == 'C':
                token_positions.append(j)
        
        # 2. Bounding boxes와 cells 정보 추출
        bboxes = []
        cells = []
        cell_idx = 0
        
        for pos in token_positions:
            if cell_idx < len(ann['html']['cells']) and ann['has_data_flags'][pos]:
                cell = ann['html']['cells'][cell_idx]
                if 'bbox' in cell:
                    x0, y0, x1, y1 = cell['bbox']
                    normalized_bbox = [x0/image_width, y0/image_height, 
                                    x1/image_width, y1/image_height]
                    bboxes.append(normalized_bbox)
                    cells.append({
                        'bbox': normalized_bbox,
                        'sequence_pos': pos  # 실제 sequence 내 위치
                    })
            cell_idx += 1
        
        if not cells:
            raise ValueError(f"No valid cells found in {image_name}")
        
        # 3. data tag set D 정의 (실제 text가 있는 C 태그)
        has_data = [False]  # BOS 토큰
        for token_id, has_text in zip(ann['otsl'], ann['has_data_flags']):
            if token_id == 'C' and has_text:
                has_data.append(True)
            else:
                has_data.append(False)
        
        # 4. box-tag 매핑 생성 (many-to-one 관계 지원)
        current_mappings = [[] for _ in range(len(bboxes))]
        for i, bbox in enumerate(bboxes):
            for cell in cells:
                if self.bbox_belongs_to_cell(bbox, cell['bbox']):
                    current_mappings[i].append(cell['sequence_pos'])
        
        # print("\nDataset __getitem__:")
        # print(f"Number of bboxes: {len(bboxes)}")
        # print(f"Number of cells: {len(cells)}")
        # print(f"Token positions: {token_positions}")
        # for i, mappings in enumerate(current_mappings):
            # print(f"Box {i} mapped to sequence positions: {mappings}")
        
        return {
            'image_name': image_name,
            'image': image,
            'otsl_str': ann['otsl'],
            'bboxes': bboxes,
            'num_boxes': len(bboxes),
            'cells': cells,
            'html': ann['html'],
            'has_data_flags': has_data[1:],
            'box_mappings': current_mappings,
            'token_positions': token_positions  # sequence 내 C 태그 위치 정보 추가
        }

    def bbox_belongs_to_cell(self, bbox1, bbox2):
        """두 bbox가 겹치는지 확인"""
        x1, y1, x2, y2 = bbox1
        cx1, cy1, cx2, cy2 = bbox2
        
        # bbox의 중심점이 cell 내부에 있는지 확인
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (cx1 <= center_x <= cx2 and 
                cy1 <= center_y <= cy2)

    def __len__(self) -> int:
        return len(self.image_names)
