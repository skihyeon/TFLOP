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
from utils.util import extract_spans_from_html, convert_html_to_otsl, extract_spans_from_otsl, compute_span_coefficients

# 디버그 모드 설정
DEBUG = True
DEBUG_SAMPLES = {
    'train': 100,
    'val': 10
}

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
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        
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
        
        # 주석 파일 로드 및 검증
        self._load_annotations()
        
    def _load_annotations(self):
        """주석 파일 로드 및 검증"""
        self.annotations = {}
        self.image_names = []
        self.cached_otsl_tokens = {}  # OTSL 토큰 캐시 추가
        self.cached_has_data = {}
        
        annotation_file = os.path.join(self.data_dir, f"{self.split}_filtered.jsonl")
        print(f"Loading {self.split} annotations from {annotation_file}")
        
        valid_count = 0
        total_count = 0
        error_counts = {}
        
        with jsonlines.open(annotation_file) as reader:
            for ann in reader:
                # DEBUG 모드일 때 샘플 수 제한
                if DEBUG and valid_count >= DEBUG_SAMPLES.get(self.split, 100):
                    break
                    
                total_count += 1
                try:
                    # 1. HTML 구조 검증
                    if 'html' not in ann or 'structure' not in ann['html']:
                        error_counts['missing_html'] = error_counts.get('missing_html', 0) + 1
                        continue
                    
                    # 2. Cell 정보 검증 - cells는 html 안에 있음
                    if 'cells' not in ann['html']:
                        error_counts['missing_cells'] = error_counts.get('missing_cells', 0) + 1
                        continue
                    
                    # 3. OTSL 변환 및 토큰화를 미리 수행
                    try:
                        otsl_sequence, has_data = convert_html_to_otsl(ann)
                        otsl_tokens = self.tokenizer.encode(otsl_sequence, html_structure=ann['html']['structure'])
                        # 검증이 완료된 토큰을 캐시에 저장
                        self.cached_otsl_tokens[ann['filename']] = otsl_tokens
                        self.cached_has_data[ann['filename']] = has_data
                    
                    except Exception as e:
                        error_counts['otsl_conversion'] = error_counts.get('otsl_conversion', 0) + 1
                        continue
                    
                    # 모든 검증을 통과한 샘플만 저장
                    self.annotations[ann['filename']] = ann
                    self.image_names.append(ann['filename'])
                    valid_count += 1
                    
                except Exception as e:
                    error_counts['other'] = error_counts.get('other', 0) + 1
                    continue
        
        # 최종 통계 출력
        print(f"\nLoading complete:")
        print(f"Total samples processed: {total_count}")
        print(f"Valid samples: {valid_count}")
        print("\nError statistics:")
        for error_type, count in error_counts.items():
            print(f"{error_type}: {count} ({count/total_count*100:.2f}%)")
        
        if valid_count == 0:
            raise ValueError(f"No valid samples found in {annotation_file}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터셋에서 하나의 샘플을 가져옴
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            Dict[str, torch.Tensor]: 처리된 데이터 샘플
            - image_name: 이미지 파일명
            - image: 정규화된 이미지 텐서 (3, H, W)
            - tokens: OTSL 토큰 시퀀스 (L)
            - bboxes: 정규화된 bbox 좌표 (N, 4)
            - box_indices: C 토큰과 bbox 매핑 (N, M)
            - data_tag_mask: C 토큰 위치 마스크 (L)
            - cells: 원본 cell 데이터
            - html: 원본 HTML 구조
            
        Raises:
            ValueError: 데이터 로드 또는 처리 중 오류 발생 시
        """
        try:
            # 1. 기본 데이터 로드 및 검증
            image_name = self.image_names[idx]
            ann = self.annotations[image_name]
            
            if 'html' not in ann or 'cells' not in ann['html'] or 'structure' not in ann['html']:
                raise ValueError(f"Invalid annotation structure for {image_name}")
            
            # 2. 이미지 로드 및 전처리
            image_path = os.path.join(self.data_dir, self.split, image_name)
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
                
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size
            image = self.transform(image)
            
            # 3. OTSL 토큰 및 has_data 플래그 검증
            if image_name not in self.cached_otsl_tokens or image_name not in self.cached_has_data:
                raise ValueError(f"Missing cached data for {image_name}")
                
            otsl_tokens = self.cached_otsl_tokens[image_name]
            has_data_flags = self.cached_has_data[image_name]
            
            # 4. Cell 정보 추출 및 정규화
            cells = []
            bboxes = []
            
            for cell in ann['html']['cells']:
                if 'bbox' not in cell:
                    continue
                    
                x0, y0, x1, y1 = cell['bbox']
                
                # bbox 유효성 검사
                if not all(isinstance(coord, (int, float)) for coord in [x0, y0, x1, y1]):
                    continue
                if x0 >= x1 or y0 >= y1:
                    continue
                if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
                    continue
                
                # bbox 정규화
                normalized_bbox = [
                    x0 / image_width,
                    y0 / image_height,
                    x1 / image_width,
                    y1 / image_height
                ]
                bboxes.append(normalized_bbox)
                
                # cell 텍스트 정제
                text = ''.join(cell['tokens'])
                text = text.replace('<b>', '').replace('</b>', '') \
                           .replace('<i>', '').replace('</i>', '') \
                           .replace('<sup>', '').replace('</sup>', '') \
                           .replace('<sub>', '').replace('</sub>', '')
                
                cells.append({
                    'text': text,
                    'bbox': normalized_bbox
                })
            
            if not bboxes:
                raise ValueError(f"No valid bboxes found in {image_name}")
            
            # 5. Box indices 및 masks 생성
            special_token_ids = {
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id
            }
            
            # Data tag mask 초기화
            data_tag_mask = torch.zeros(len(otsl_tokens), dtype=torch.bool)
            
            # C 토큰 위치와 데이터 존재 여부 매핑
            valid_c_positions = []  # 실제 데이터가 있는 'C' 토큰의 위치
            all_c_positions = []    # 모든 'C' 토큰의 위치
            
            has_data_flags = [False] + has_data_flags  # [BOS] 토큰을 위한 패딩
            
            # C 토큰 위치 계산
            c_token_id = self.tokenizer.token2id["C"]
            for i, (token_id, has_data) in enumerate(zip(otsl_tokens, has_data_flags)):
                if token_id in special_token_ids:
                    continue
                    
                if token_id == c_token_id:
                    data_tag_mask[i] = True
                    all_c_positions.append(i)
                    if has_data:
                        valid_c_positions.append(i)
            
            # Box indices 계산
            box_to_tokens = [[] for _ in range(len(bboxes))]
            
            # HTML cells의 순서대로 매핑
            cell_idx = 0
            for pos in all_c_positions:
                if cell_idx >= len(bboxes):
                    break
                if pos in valid_c_positions:
                    box_to_tokens[cell_idx].append(pos)
                cell_idx += 1
            
            # 최대 매핑 수 계산 및 padding
            max_mappings = max(max(len(tokens) for tokens in box_to_tokens), 1)
            box_indices = torch.full((len(bboxes), max_mappings), -1, dtype=torch.long)
            
            for i, tokens in enumerate(box_to_tokens):
                if tokens:
                    box_indices[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            
            # 6. 최종 결과 반환
            return {
                'image_name': image_name,
                'image': image,                                                    # (3, H, W)
                'tokens': torch.tensor(otsl_tokens, dtype=torch.long),            # (L)
                'bboxes': torch.tensor(bboxes, dtype=torch.float32),             # (N, 4)
                'box_indices': box_indices,                                       # (N, M)
                'data_tag_mask': data_tag_mask,                                  # (L)
                'cells': cells,
                'html': ann['html']
            }
            
        except Exception as e:
            print(f"Error processing sample {idx} ({image_name if 'image_name' in locals() else 'unknown'}): {str(e)}")
            # 다음 유효한 샘플 반환
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self) -> int:
        return len(self.image_names)

if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from utils.visualize import visualize_validation_sample
    from utils.util import construct_table_html_gt
    from pathlib import Path
    
    # 데이터셋 로드
    dataset = TableDataset(data_dir="./data/pubtabnet", split="val")
    
    otsl_str = ""
        # 랜덤 샘플 선택
    random_idx = random.randint(0, len(dataset) - 1)
    # random_idx = 1
    sample = dataset[random_idx]

    # OTSL 토큰을 문자열로 변환
    otsl_str = ' '.join([dataset.tokenizer.id2token[tid.item()] for tid in sample['tokens']])

    # HTML 문자열 추출
    html_str = str(sample['html'])
    print("html_structure: ", html_str)
    print("cells order: ", [cell['text'] for cell in sample['cells']])
    print("sample['image_name']: ", sample['image_name'])

    print("box indices: ", sample['box_indices'])
    # 디버그 디렉토리 생성
    debug_viz_dir = Path("./debug_viz")
    os.makedirs(debug_viz_dir, exist_ok=True)
    os.makedirs(debug_viz_dir / "html_tables", exist_ok=True)
    
    # GT HTML 테이블 생성 및 저장
    text_regions = [
        {
            'text': cell['text'],
            'bbox': cell['bbox']
        }
        for cell in sample['cells']
    ]
    
    gt_table_html = construct_table_html_gt(
        sample['html']
    )
    
    # HTML 파일로 저장 (스타일 포함)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <h2>Ground Truth Table (Sample {random_idx})</h2>
        {gt_table_html}
    </body>
    </html>
    """
    
    html_path = debug_viz_dir / "html_tables" / f"table_{random_idx:08d}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 시각화 함수 호출
    visualize_validation_sample(
        image=sample['image'],
        boxes=sample['bboxes'],
        pred_html="",  # 예측값은 없으므로 빈 문자열
        true_html=html_str,  # 너무 길지 않게 잘라서 표시
        pred_otsl="",  # 예측값은 없으므로 빈 문자열
        true_otsl=otsl_str,
        pointer_logits=None,  # 예측값이 없으므로 None
        step=random_idx,  # 파일명 용도로 랜덤 인덱스 사용
        viz_dir=debug_viz_dir  # 현재 경로 아래 debug_viz 폴더에 저장
    )
    
    print(f"\n랜덤 샘플 {random_idx} 시각화 완료")
    print(f"이미지 경로: ./debug_viz/images/val_step_{random_idx:08d}.png")
    print(f"텍스트 정보: ./debug_viz/txts/val_step_{random_idx:08d}.txt")
    print(f"HTML 테이블: ./debug_viz/html_tables/table_{random_idx:08d}.html")
    print("\nOTSL 시퀀스 예시:")
    print(otsl_str[:200], "...")