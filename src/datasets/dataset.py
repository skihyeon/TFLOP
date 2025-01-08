import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from PIL import Image
from torch.utils.data import Dataset
from models.otsl_tokenizer import OTSLTokenizer
from typing import Dict, Optional, Any
from utils.util import convert_html_to_otsl
from transformers import AutoImageProcessor
from config import ModelConfig
from pathlib import Path
import mmap
import json
from collections import Counter

# 디버그 모드 설정
DEBUG = False
DEBUG_SAMPLES = {
    'train': 1000,
    'val': 100
}

class BaseTableDataset(Dataset):
    """기본 테이블 데이터셋 클래스"""
    def __init__(
        self,
        data_dir: str,
        image_size: int = 768,
        model_config: Optional[Any] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.config = model_config
        
        # 이미지 프로세서 초기화 - config에서 model_name 가져오기
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.swin_model_name if self.config else ModelConfig.swin_model_name
        )
        self.image_processor.size = (image_size, image_size)
    
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

class TableDataset(BaseTableDataset):
    """
    테이블 이미지 데이터셋 (학습/평가용)
    
    논문 Section 4.1 Datasets 참조:
    - PubTabNet, FinTabNet, SynthTabNet 데이터셋 지원
    - cell-level annotations 또는 OCR engine으로부터 text regions 획득
    - layout prompt를 위한 sequence length 및 position 조정 포함
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 768,
        tokenizer: OTSLTokenizer = None,
        model_config: Optional[Any] = None,
    ):
        super().__init__(data_dir, image_size, model_config)
        self.split = split
        self.tokenizer = tokenizer
        self.layout_prompt_length = self.config.total_sequence_length - self.config.otsl_max_length
        
        self._load_and_filter_annotations()
    
    def _load_and_filter_annotations(self):
        """주석 파일 로드 및 sequence length 기준으로 필터링"""
        ann_file = os.path.join(self.data_dir, f'{self.split}.jsonl')
        
        self.annotations = {}
        self.image_names = []
        filtered_count = 0
        token_counts = Counter()
        
        with open(ann_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            while True:
                line = mm.readline()
                if not line:  # EOF
                    break
                    
                if DEBUG and len(self.image_names) >= DEBUG_SAMPLES[self.split]:
                    break
                    
                try:
                    ann = json.loads(line.decode('utf-8'))
                    otsl_tokens_list, has_data_flags_list = convert_html_to_otsl(ann)
                    
                    # OTSL 시퀀스 유효성 검사
                    if (otsl_tokens_list is None or 
                        not self.tokenizer.validate_syntax(otsl_tokens_list)):
                        filtered_count += 1
                        continue
                    
                    # 길이 검증
                    num_boxes = len(ann['html']['cells'])
                    otsl_length = len(otsl_tokens_list)
                    
                    if (num_boxes > self.layout_prompt_length or
                        otsl_length > self.config.otsl_max_length - 1):
                        filtered_count += 1
                        continue
                    
                    # C 토큰과 HTML cells 핑 검증
                    token_positions = [j for j, token in enumerate(otsl_tokens_list) if token == 'C']
                    if len(token_positions) != len(ann['html']['cells']):
                        filtered_count += 1
                        continue
                    
                    # 토큰 카운트 업데이트
                    token_counts.update(otsl_tokens_list)
                    
                    image_name = ann['filename']
                    # 필요한 정보만 저장
                    self.annotations[image_name] = {
                        'html': ann['html'],
                        'otsl_tokens_list': otsl_tokens_list,
                        'has_data_flags_list': has_data_flags_list
                    }
                    self.image_names.append(image_name)
                    
                except Exception as e:
                    filtered_count += 1
                    if DEBUG:
                        print(f"Error processing annotation: {str(e)}")
                    continue
            
        print(f"\n=== Dataset Loading Summary ===")
        print(f"Total loaded: {len(self.image_names)}")
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
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        image = self.image_processor(image, return_tensors="pt")
        
        # C 태그 위치 추적 및 데이터 처리
        cells = []
        bboxes = []
        bbox_with_text = {}
        current_mappings = {}  # cell_idx -> bbox_idx list 매핑

        # C 토큰 위치 가져오기
        token_positions = [j for j, token in enumerate(ann['otsl_tokens_list']) if token == 'C']
        
        # cells, bboxes, text 정보 �리
        for cell_idx, (pos, cell) in enumerate(zip(token_positions, ann['html']['cells'])):
            cell_info = {
                'sequence_pos': pos,
                'cell_idx': cell_idx,
                'has_data': False
            }
            
            if 'bbox' in cell:
                # bbox 정규화
                x0, y0, x1, y1 = cell['bbox']
                normalized_bbox = [
                    x0/image_width, y0/image_height,
                    x1/image_width, y1/image_height
                ]
                
                # cell 정보 업데이트
                cell_info.update({
                    'bbox': normalized_bbox,
                    'has_data': True
                })
                bboxes.append(normalized_bbox)
                
                # bbox 매핑 생성
                current_mappings[cell_idx] = [len(bboxes) - 1]
                
                # text 정보 저장
                tokens = cell.get('tokens', [])
                bbox_with_text[cell_idx] = {
                    'bbox': cell['bbox'],
                    'text': self.tokens_to_text(tokens) if tokens else ""
                }
            
            cells.append(cell_info)
        
        return {
            'image_name': image_name,
            'image': image,
            'otsl_tokens_list': ann['otsl_tokens_list'],
            'bboxes': bboxes,
            'num_boxes': len(bboxes),
            'cells': cells,
            'html': ann['html'],
            'box_mappings': current_mappings,
            'bbox_with_text': bbox_with_text
        }

    def __len__(self) -> int:
        return len(self.annotations)

class PredictTableDataset(BaseTableDataset):
    """예측용 테이블 데이터셋"""
    def __init__(
        self,
        data_dir: str,
        image_size: int = 768,
        model_config: Optional[Any] = None,  # config 추가
        ocr_results: Optional[Dict] = None
    ):
        super().__init__(data_dir, image_size, model_config)  # config 전달
        self.ocr_results = ocr_results
        
        # 이미지 경로만 수집
        self.image_paths = list(Path(data_dir).glob("*.[jp][pn][g]"))
        self.image_names = [path.name for path in self.image_paths]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """예측용 데이터 반환"""
        image_path = self.image_paths[idx]
        image_name = image_path.name
        
        # 1. 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        image = self.image_processor(image, return_tensors="pt")
        
        # 2. OCR 결과 처리
        boxes = []
        bbox_with_text = {}
        
        if self.ocr_results and image_name in self.ocr_results:
            for i, item in enumerate(self.ocr_results[image_name]):
                x0, y0, x1, y1 = item['bbox']
                normalized_bbox = [
                    x0/image_width, y0/image_height,
                    x1/image_width, y1/image_height
                ]
                boxes.append(normalized_bbox)
                bbox_with_text[i] = {
                    'bbox': normalized_bbox,
                    'text': item['text']
                }
        
        return {
            'image_name': image_name,
            'images': image['pixel_values'].squeeze(0),
            'bboxes': torch.FloatTensor(boxes) if boxes else torch.zeros((0, 4)),
            'bbox_with_text': bbox_with_text,
            'num_boxes': len(boxes)
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)
