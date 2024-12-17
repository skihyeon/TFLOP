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
from transformers import AutoImageProcessor
from config import ModelConfig

# 디버그 모드 설정
DEBUG = True
DEBUG_SAMPLES = {
    'train': 1280,
    'val': 64
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
        self.config = ModelConfig()
        # 고정된 길이 설정
        self.layout_prompt_length = self.config.total_sequence_length - self.config.otsl_max_length
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.image_processor = AutoImageProcessor.from_pretrained(ModelConfig.swin_model_name)
        self.image_processor.size = (image_size, image_size)
        
        # 주석 파일 로드 및 필터링
        self._load_and_filter_annotations()
        
    def _load_and_filter_annotations(self):
        """주석 파일 로드 및 sequence length 기준으로 필터링"""
        ann_file = os.path.join(self.data_dir, f'{self.split}_filtered.jsonl')
        
        self.annotations = {}
        self.image_names = []
        filtered_count = 0
        
        # 토큰 통계를 위한 카운터 초기화
        token_counts = {
            '[BOS]': 0,
            '[EOS]': 0,
            'C': 0, 
            'NL': 0,
            'L': 0,
            'U': 0,
            'X': 0
        }
        
        with jsonlines.open(ann_file) as reader:
            for ann in reader:
                if DEBUG and len(self.image_names) >= DEBUG_SAMPLES[self.split]:
                    break
                    
                try:
                    otsl_tokens_list, has_data_flags_list = convert_html_to_otsl(ann)
                    if otsl_tokens_list is None:
                        filtered_count += 1
                        continue
                    
                    # 길이 검증
                    num_boxes = len(ann['html']['cells'])
                    otsl_length = len(otsl_tokens_list)
                    
                    # layout prompt와 otsl sequence 각각의 길이 제한 검증 && otsl sequence에서 BOS, EOS 토큰 제외
                    if (num_boxes > self.layout_prompt_length -2 or 
                        otsl_length > self.config.otsl_max_length - 2):
                        filtered_count += 1
                        continue
                    
                    # 토큰 카운트 업데이트
                    token_counts['[BOS]'] += 1
                    token_counts['[EOS]'] += 1
                    for token in otsl_tokens_list:
                        if token in token_counts:
                            token_counts[token] += 1
                    
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
        
        # 토큰 분포 출력
        print("\n=== Token Distribution Debug ===")
        print("Token Distributions:")
        for token, count in token_counts.items():
            print(f"  {token:<7}: {count:>4}")
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
        # image = self.transform(image)
        image = self.image_processor(image, return_tensors="pt")
        
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
            if cell_idx >= len(ann['html']['cells']):
                break
            
            if ann['has_data_flags_list'][pos]:
                cell = ann['html']['cells'][cell_idx]
                if 'bbox' in cell:
                    x0, y0, x1, y1 = cell['bbox']
                    normalized_bbox = [x0/image_width, y0/image_height, 
                                    x1/image_width, y1/image_height]
                    bboxes.append(normalized_bbox)
                    cells.append({
                        'bbox': normalized_bbox,
                        'sequence_pos': pos,
                        'cell_idx': cell_idx
                    })
            cell_idx += 1
            
        
        if not cells:
            raise ValueError(f"No valid cells found in {image_name}")
        
        # 3. data tag set D 정의 수정
        has_data = []
        has_data.append(False)  # BOS
        for token_id, has_text in zip(ann['otsl_tokens_list'], ann['has_data_flags_list']):
            if token_id == 'C' and has_text:
                has_data.append(True)
            else:
                has_data.append(False)
        has_data.append(False)  # EOS
        
        # padding을 위해 최대 길이까지 False로 채우기
        while len(has_data) < self.config.otsl_max_length:
            has_data.append(False)
        
        # 4. box-tag 매핑 생성 (many-to-one 관계 지원)
        current_mappings = self._create_box_tag_mappings(bboxes, cells, ann['otsl_tokens_list'])
        
        bbox_with_text = {}
        for i, cell in enumerate(ann['html']['cells']):
            tokens = cell.get('tokens', [])  # tokens가 없을 경우 빈 리스트 반환
            bbox = cell.get('bbox')          # bbox가 없을 수 있음
            
            if 'tokens' in cell:  # tokens 키가 존재하면 (빈 리스트여도) 처리
                text = self.tokens_to_text(tokens) if tokens else ""
                bbox_with_text[i] = {
                    'bbox': bbox,  # bbox가 None일 수 있음
                    'text': text
                }
                    

        
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
            'token_positions': token_positions,        # sequence 내 C 태그 위치 정보 추가
            'bbox_with_text': bbox_with_text
        }

    def bbox_belongs_to_cell(self, bbox1, bbox2):
        """두 bbox의 IoU나 overlap 비율로 판단"""
        def compute_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        
        # intersection 계산
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = compute_area(bbox1)
        
        # bbox1의 50% 이상이 cell 내부에 있어야 함
        return intersection / bbox1_area > 0.5
    
    def tokens_to_text(self, tokens):
        # HTML 태그 제거 (<b>, </b> 등)
        text_tokens = [token for token in tokens if not (token.startswith('<') and token.endswith('>'))]
        # 토큰들을 하나의 문자열로 합치기
        text = ''.join(text_tokens)
        return text


    
    def _create_box_tag_mappings(self, bboxes, cells, otsl_tokens_list):
        """박스와 태그 간의 매핑을 생성"""
        current_mappings = [[] for _ in range(len(bboxes))]
        cell_idx = 0
        
        # OTSL sequence에서 C 태그의 위치를 추적하되, span 정보도 함께 저장
        token_to_cell_idx = {}  # sequence position -> cell_idx 매핑
        current_cell_idx = 0
        
        for i, token in enumerate(otsl_tokens_list):
            if token == 'C':
                token_to_cell_idx[i] = current_cell_idx
                current_cell_idx += 1
            elif token in ['L', 'U', 'X']:  # span 태그들도 cell_idx에 매핑
                token_to_cell_idx[i] = current_cell_idx - 1  # 이전 셀에 연결
        
        # 각 bbox에 대해 매핑 생성
        for i, bbox in enumerate(bboxes):
            for cell in cells:
                if self.bbox_belongs_to_cell(bbox, cell['bbox']):
                    sequence_pos = cell['sequence_pos']
                    if sequence_pos < self.config.otsl_max_length:
                        # span된 셀의 경우 관련된 모든 position을 매핑에 추가
                        for pos, idx in token_to_cell_idx.items():
                            if idx == cell_idx:
                                current_mappings[i].append(pos)
                    cell_idx += 1
        
        # Debug information
        # print(f"\n=== Box-Tag Mapping Debug ===")
        # print(f"Number of boxes: {len(bboxes)}")
        # print(f"Number of cells: {len(cells)}")
        # print(f"OTSL sequence: {' '.join(otsl_tokens_list)}")
        # print(f"Token to cell mapping: {token_to_cell_idx}")
        # for i, mapping in enumerate(current_mappings):
        #     if mapping:
        #         print(f"Box {i}: mapped to positions {mapping}")
        #     else:
        #         print(f"Box {i}: no mapping")
        
        return current_mappings


    def __len__(self) -> int:
        return len(self.image_names)
    