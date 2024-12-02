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
    'train': 10000,
    'val': 100
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
        
        # 주석 파일 로드 및 검증
        self._load_annotations()
        
    def _load_annotations(self):
        """주석 파일 로드 및 검증"""
        self.annotations = {}
        self.image_names = []
        self.cached_otsl_tokens = {}  # OTSL 토큰 캐시 추가
        
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
                        otsl_sequence = convert_html_to_otsl(ann)
                        otsl_tokens = self.tokenizer.encode(otsl_sequence, html_structure=ann['html']['structure'])
                        # 검증이 완료된 토큰을 캐시에 저장
                        self.cached_otsl_tokens[ann['filename']] = otsl_tokens
                    
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

    def __getitem__(self, idx: int) -> Dict:
        """데이터셋에서 하나의 샘플을 가져옴"""
        try:
            image_name = self.image_names[idx]
            ann = self.annotations[image_name]
            
            # 1. 이미지 로드 및 전처리
            image = Image.open(os.path.join(self.data_dir, self.split, image_name)).convert('RGB')
            image_width, image_height = image.size
            image = self.transform(image)
            
            # 2. 캐시된 OTSL 토큰 사용
            otsl_tokens = self.cached_otsl_tokens[image_name]
            
            # 4. Cell 정보 추출 및 정규화
            cells = []
            bboxes = []
            for cell in ann['html']['cells']:
                if 'bbox' in cell:
                    x0, y0, x1, y1 = cell['bbox']
                    normalized_bbox = [
                        x0 / image_width,
                        y0 / image_height,
                        x1 / image_width,
                        y1 / image_height
                    ]
                    bboxes.append(normalized_bbox)
                    
                    # tokens를 하나의 텍스트로 합치기
                    text = ''.join(cell['tokens'])
                    # HTML 태그 제거 (선택사항)
                    text = text.replace('<b>', '').replace('</b>', '') \
                             .replace('<i>', '').replace('</i>', '') \
                             .replace('<sup>', '').replace('</sup>', '')
                    
                    cells.append({
                        'text': text,
                        'bbox': normalized_bbox
                    })
            
            special_token_ids = {
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id
            }
            
            # Data tag mask와 box indices 생성
            data_tag_positions = []  # 'C' 토큰의 위치
            box_indices = []         # bbox가 있는 'C' 토큰의 위치
            empty_mask = torch.zeros(len(otsl_tokens), dtype=torch.bool)  # empty cell mask
            
            # 1. 모든 'C' 토큰의 위치 찾기
            for i, token_id in enumerate(otsl_tokens):
                if token_id in special_token_ids:
                    continue
                token = self.tokenizer.id2token[token_id]
                if token == 'C':
                    data_tag_positions.append(i)
            
            # 2. Data tag mask 생성 - 모든 'C' 토큰 위치를 True로
            data_tag_mask = torch.zeros(len(otsl_tokens), dtype=torch.bool)
            for pos in data_tag_positions:
                data_tag_mask[pos] = True
            
            # 3. Box indices와 empty mask 생성
            bbox_to_token_map = {}  # bbox index -> token position
            curr_bbox_idx = 0
            
            # 먼저 bbox가 있는 cell 매핑
            for cell in ann['html']['cells']:
                if 'bbox' in cell:  # bbox가 있는 cell
                    if curr_bbox_idx < len(data_tag_positions):
                        bbox_to_token_map[curr_bbox_idx] = data_tag_positions[curr_bbox_idx]
                        curr_bbox_idx += 1
            
            # Empty cell 찾기 (bbox가 없는 'C' 토큰)
            mapped_positions = set(bbox_to_token_map.values())
            for pos in data_tag_positions:
                if pos not in mapped_positions:
                    empty_mask[pos] = True
            
            # Box indices 텐서 생성
            box_indices = []
            for i in range(len(bboxes)):
                if i in bbox_to_token_map:
                    box_indices.append(bbox_to_token_map[i])
                else:
                    box_indices.append(-1)  # padding index
            
            return {
                'image_name': image_name,
                'image': image,
                'tokens': torch.tensor(otsl_tokens, dtype=torch.long),
                'bboxes': torch.tensor(bboxes, dtype=torch.float32),
                'box_indices': torch.tensor(box_indices, dtype=torch.long),
                'data_tag_mask': data_tag_mask,
                'empty_mask': empty_mask,  # 추가: empty cell mask
                'cells': cells,
                'html': ann['html']
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
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