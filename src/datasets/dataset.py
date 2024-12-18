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

            
        # 1. 모든 셀에 대한 정보 수집
        cells = []
        bboxes = []
        cell_idx = 0
        
        for pos in token_positions:
            if cell_idx >= len(ann['html']['cells']):
                break
                
            cell = ann['html']['cells'][cell_idx]
            cell_info = {
                'sequence_pos': pos,
                'cell_idx': cell_idx,
                'has_data': ann['has_data_flags_list'][pos]
            }
            
            # bbox가 있는 경우 추가
            if 'bbox' in cell:
                x0, y0, x1, y1 = cell['bbox']
                normalized_bbox = [x0/image_width, y0/image_height, 
                                x1/image_width, y1/image_height]
                cell_info['bbox'] = normalized_bbox
                bboxes.append(normalized_bbox)
                
            cells.append(cell_info)
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
        
        # box-tag 매핑 생성
        current_mappings = self._create_box_tag_mappings(bboxes, cells, ann['otsl_tokens_list'])
        
        # 매핑 검증 및 보정
        for bbox_idx in range(len(bboxes)):
            mapped = False
            for cell_mappings in current_mappings.values():
                if bbox_idx in cell_mappings:
                    mapped = True
                    break
            
            if not mapped:
                # bbox가 있는 첫 번째 cell을 찾아서 매핑
                for cell in cells:
                    if 'bbox' in cell:
                        current_mappings[cell['cell_idx']].append(bbox_idx)
                        print(f"Warning: Sample {idx}, Box {bbox_idx} mapped to default cell {cell['cell_idx']}")
                        break
        
        
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

    def compute_overlap_ratio(self, bbox1, bbox2):
        """두 bbox 간의 겹침 비율 계산"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        
        return intersection / bbox1_area
    
    def tokens_to_text(self, tokens):
        # HTML 태그 제거 (<b>, </b> 등)
        text_tokens = [token for token in tokens if not (token.startswith('<') and token.endswith('>'))]
        # 토큰들을 하나의 문자열로 합치기
        text = ''.join(text_tokens)
        return text


    
    def _create_box_tag_mappings(self, bboxes, cells, otsl_tokens_list):
        """박스와 태그 간의 매핑을 생성"""
        # # 디버그 정보를 파일에 저장
        # with open('debug.txt', 'a') as f:
        #     f.write("\n=== Debug: Box Tag Mapping ===\n")
        #     f.write(f"Number of bboxes: {len(bboxes)}\n")
        #     f.write(f"Number of cells: {len(cells)}\n") 
        #     f.write(f"OTSL sequence length: {len(otsl_tokens_list)}\n")
            
        #     f.write("\nOTSL Sequence:\n")
        #     f.write(" ".join(otsl_tokens_list) + "\n")
            
        #     f.write("\nCell Information:\n")
        #     for cell in cells:
        #         f.write(f"Cell {cell['cell_idx']} at sequence pos {cell['sequence_pos']}:\n")
        #         # bbox가 있는 경우에만 출력
        #         if 'bbox' in cell:
        #             f.write(f"  bbox: {cell['bbox']}\n")
        #         else:
        #             f.write("  bbox: None (empty cell)\n")
            
        #     f.write("\nBBox Information:\n")
        #     for i, bbox in enumerate(bboxes):
        #         f.write(f"Box {i}: {bbox}\n")
        
        # cells의 sequence_pos -> cell_idx 매핑 생성 (bbox가 있는 셀만)
        seq_pos_to_cell_idx = {
            cell['sequence_pos']: cell['cell_idx'] 
            for cell in cells 
            if 'bbox' in cell
        }
        
        # bbox가 있는 셀에 대해서만 매핑 생성
        current_mappings = {
            cell['cell_idx']: [] 
            for cell in cells 
            if 'bbox' in cell
        }
        
        token_to_cell_idx = {}
        current_cell_idx = 0
        current_row_start_idx = 0
        prev_row_start_idx = 0
        
        # OTSL 시퀀스 처리 - span 태그 포함
        # with open('debug.txt', 'a') as f:
        #     f.write("\nToken to Cell Index Mapping:\n")
        for i, token in enumerate(otsl_tokens_list):
            if token == 'C':
                # 현재 position이 실제 cell과 매핑되는지 확인
                if i in seq_pos_to_cell_idx:
                    token_to_cell_idx[i] = seq_pos_to_cell_idx[i]
                current_cell_idx += 1
                # f.write(f"Token {i} (C) -> Cell {current_cell_idx-1}\n")
            elif token == 'L':
                if current_cell_idx > current_row_start_idx:
                    token_to_cell_idx[i] = current_cell_idx - 1
                    # f.write(f"Token {i} (L) -> Cell {current_cell_idx - 1}\n")
            elif token == 'U':
                cells_in_prev_row = current_row_start_idx - prev_row_start_idx
                relative_pos = (current_cell_idx - current_row_start_idx)
                if cells_in_prev_row > relative_pos:
                    mapped_cell = prev_row_start_idx + relative_pos
                    token_to_cell_idx[i] = mapped_cell
                    # f.write(f"Token {i} (U) -> Cell {mapped_cell}\n")
            elif token == 'X':
                if current_cell_idx > current_row_start_idx:
                    left_cell_idx = current_cell_idx - 1
                    cells_in_prev_row = current_row_start_idx - prev_row_start_idx
                    relative_pos = (current_cell_idx - current_row_start_idx)
                    if cells_in_prev_row > relative_pos:
                        up_cell_idx = prev_row_start_idx + relative_pos
                        mapped_cell = min(left_cell_idx, up_cell_idx)
                        token_to_cell_idx[i] = mapped_cell
                        # f.write(f"Token {i} (X) -> Cell {mapped_cell}\n")
            elif token == 'NL':
                prev_row_start_idx = current_row_start_idx
                current_row_start_idx = current_cell_idx
                # f.write(f"Token {i} (NL) -> Row break (next row starts at cell {current_cell_idx})\n")
            
        # f.write("\nOverlap Calculations:\n")
        for bbox_idx, bbox in enumerate(bboxes):
            # f.write(f"\nChecking Box {bbox_idx}:\n")
            mapped = False
            best_overlap = 0
            best_cell_idx = None
            
            for cell in cells:
                if 'bbox' not in cell:  # bbox가 없는 빈 셀은 건너뛰기
                    continue

                # f.write(f"  With Cell {cell['cell_idx']}: overlap = {overlap_ratio:.3f}\n")
                overlap_ratio = self.compute_overlap_ratio(bbox, cell['bbox'])
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cell_idx = cell['cell_idx']
                    mapped_cell_idx = token_to_cell_idx.get(cell['sequence_pos'], -1)
                    if 0 <= mapped_cell_idx < len(cells):
                        mapped = True
            
            if mapped and best_overlap > 0.5:
                current_mappings[best_cell_idx].append(bbox_idx)
                # f.write(f"  -> Mapped to Cell {best_cell_idx} (overlap: {best_overlap:.3f})\n")
            # else:
            #     f.write(f"  -> No valid mapping (best overlap: {best_overlap:.3f} with Cell {best_cell_idx})\n")
            
        # f.write("\nFinal Mappings:\n")
        # for cell_idx, box_indices in current_mappings.items():
        #     f.write(f"Cell {cell_idx} -> Boxes: {box_indices}\n")
        
        return current_mappings

    def __len__(self) -> int:
        return len(self.image_names)
    