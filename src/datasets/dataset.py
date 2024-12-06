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
import numpy as np

# 디버그 모드 설정
DEBUG = True
DEBUG_SAMPLES = {
    'train': 100000,
    'val': 100
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
        image_size: int = 768,       # 논문 4.2,
        tokenizer: OTSLTokenizer = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        
        # 고정된 길이 설정
        self.layout_prompt_length = self.otsl_sequence_length = tokenizer.otsl_sequence_length
        
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
        
        self.annotations = {}
        self.image_names = []
        filtered_count = 0
        
        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                if DEBUG and len(self.image_names) >= DEBUG_SAMPLES[self.split]:
                    break
                    
                try:
                    otsl_tokens_list, has_data_flags_list = convert_html_to_otsl(ann)
                    if otsl_tokens_list is None:
                        filtered_count += 1
                        continue
                    ## for debug
                    # if len(otsl_tokens_list) > 30:
                    #     filtered_count += 1
                    #     continue
                    # if 'U' not in otsl_tokens_list:
                    #     filtered_count += 1
                    #     continue
                    ## 
                    
                    # 길이 검증
                    num_boxes = len(ann['html']['cells'])
                    otsl_length = len(otsl_tokens_list)
                    
                    # layout prompt와 otsl sequence 각각의 길이 제한 검증 && otsl sequence에서 BOS, EOS 토큰 제외
                    if (num_boxes > self.layout_prompt_length -2 or 
                        otsl_length > self.otsl_sequence_length - 2):
                        filtered_count += 1
                        continue
                    
                    image_name = ann['filename']
                    self.annotations[image_name] = {
                        'html': ann['html'],
                        'otsl_tokens_list': otsl_tokens_list,
                        'has_data_flags_list': has_data_flags_list
                    }
                    self.image_names.append(image_name)
                    
                except Exception:
                    filtered_count += 1
                    continue
        
        print(f"\nDataset {self.split}:")
        print(f"Valid samples: {len(self.image_names)}")
        print(f"Filtered: {filtered_count}")
    
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
        for j, token in enumerate(ann['otsl_tokens_list']):
            if token == 'C':
                token_positions.append(j) 
        # 2. Bounding boxes와 cells 정보 추출
        bboxes = []
        cells = []
        cell_idx = 0
        
        for pos in token_positions:
            if cell_idx < len(ann['html']['cells']) and ann['has_data_flags_list'][pos]:
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
        for token_id, has_text in zip(ann['otsl_tokens_list'], ann['has_data_flags_list']):
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
            'image': image,                            # (3, 1024, 1024)
            'otsl_tokens_list': ann['otsl_tokens_list'], # (S)
            'bboxes': bboxes,                          # (N, 4)
            'num_boxes': len(bboxes),                  # N
            'cells': cells,
            'html': ann['html'],
            'has_data_flags_list': has_data,           # (S)
            'box_mappings': current_mappings,          # (N, max_mappings)
            'token_positions': token_positions  # sequence 내 C 태그 위치 정보 추가
        }

    def bbox_belongs_to_cell(self, bbox1, bbox2):
        """두 bbox가 치는지 확인"""
        x1, y1, x2, y2 = bbox1
        cx1, cy1, cx2, cy2 = bbox2
        
        # bbox의 중심점이 cell 내부에 있는지 확인
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (cx1 <= center_x <= cx2 and 
                cy1 <= center_y <= cy2)

    def __len__(self) -> int:
        return len(self.image_names)
    