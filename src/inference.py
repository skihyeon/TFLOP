import pytorch_lightning as pl
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import shutil
from datetime import datetime
import os
from transformers import AutoImageProcessor
from torch.utils.data import Dataset, DataLoader

from models.tflop import TFLOP
from models.otsl_tokenizer import OTSLTokenizer

class InferenceDataset(Dataset):
    """Inference용 데이터셋"""
    def __init__(
        self,
        image_paths: List[Path],
        image_processor: AutoImageProcessor,
        ocr_results: Optional[Dict[str, List[Dict[str, Union[List[float], str]]]]] = None
    ):
        self.image_paths = image_paths
        self.image_processor = image_processor
        self.ocr_results = ocr_results or {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        processed_image = self.image_processor(image, return_tensors="pt")
        
        # OCR 결과 처리
        image_ocr = self.ocr_results.get(str(image_path), [])
        
        return {
            'image_path': image_path,
            'image': processed_image,
            'ocr_results': image_ocr
        }

def inference_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Inference용 collate function"""
    return {
        'image_paths': [item['image_path'] for item in batch],
        'images': torch.cat([item['image']['pixel_values'] for item in batch], dim=0),
        'ocr_results': [item['ocr_results'] for item in batch]
    }

def create_inference_dataloader(
    image_dir: Union[str, Path],
    image_processor: AutoImageProcessor,
    ocr_results: Optional[Dict[str, List[Dict[str, Union[List[float], str]]]]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Inference용 데이터로더 생성"""
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'}
    
    # 이미지 파일 목록 생성
    image_paths = [
        p for p in image_dir.iterdir() 
        if p.suffix in image_extensions
    ]
    
    # 데이터셋 생성
    dataset = InferenceDataset(
        image_paths=image_paths,
        image_processor=image_processor,
        ocr_results=ocr_results
    )
    
    # 데이터로더 생성
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # inference에서는 순서 유지
        num_workers=num_workers,
        collate_fn=inference_collate_fn,
        pin_memory=pin_memory
    )

class TFLOPInferenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # TFLOP 모델 초기화
        self.model = TFLOP(config, inference_mode=True)
        
        # 토크나이저 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=config.total_sequence_length // 2
        )
        
        # 이미지 전처리기 초기화
        self.image_processor = AutoImageProcessor.from_pretrained(config.swin_model_name)
        self.image_processor.size = (config.image_size, config.image_size)

    def setup(self, stage=None):
        """모델 가중치 로드"""
        if hasattr(self.config, 'checkpoint_path'):
            state_dict = torch.load(self.config.checkpoint_path, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict)

    def predict_step(self, batch, batch_idx):
        """배치 단위 추론"""
        images = batch['images']
        ocr_results = batch['ocr_results']
        batch_size = len(images)

        # OCR 결과 처리
        padded_bboxes = torch.zeros(
            batch_size, 
            self.config.total_sequence_length // 2, 
            4, 
            device=self.device
        )
        
        bbox_with_text_list = []
        for i, ocr in enumerate(ocr_results):
            if ocr:  # OCR 결과가 있는 경우
                bboxes = torch.tensor([r['bbox'] for r in ocr], device=self.device)
                padded_bboxes[i, :len(ocr)] = bboxes
                bbox_with_text = {
                    j: {'bbox': r['bbox'], 'text': r['text']} 
                    for j, r in enumerate(ocr)
                }
            else:
                bbox_with_text = {}
            bbox_with_text_list.append(bbox_with_text)

        attention_mask = torch.zeros(batch_size, self.config.total_sequence_length, dtype=torch.bool).to(self.device)
        attention_mask[:, :self.config.total_sequence_length//2 ] = True  # layout prompt 부분만 True로 설정
        
        # 모델 추론
        outputs = self.model({
            'images': images,
            'bboxes': padded_bboxes,
            'attention_mask': attention_mask
        })

        # 결과 처리
        results = []
        for i in range(batch_size):
            # pred_tokens = outputs['tag_logits'][i].argmax(dim=-1)
            pred_tokens = outputs['generated_ids'][i]
            
            pred_otsl = self.tokenizer.decode(pred_tokens.cpu().tolist())
            print(f"Decoded OTSL: {pred_otsl}")
            
            print("-" * 50)
            
            result = {
                'otsl_sequence': pred_otsl,
                'pointer_logits': outputs['pointer_logits'][i],
                'empty_pointer_logits': outputs['empty_pointer_logits'][i]
            }
            
            # HTML 변환
            html = construct_table_html_pred(
                pred_otsl,
                bbox_with_text_list[i],
                outputs['pointer_logits'][i]
            )
            result['html'] = html
            
            results.append(result)
            
        return results

def process_directory(config):
    """디렉토리 단위 처리"""
    # 입력 이미지 폴더 확인
    image_dir = Path(config.image_path)
    if not image_dir.exists() or not image_dir.is_dir():
        raise ValueError(f"Invalid image directory: {image_dir}")
    
    # 출력 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_path) / f"{timestamp}_result"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # HTML 및 이미지 저장 폴더 생성
    html_dir = output_dir / "html"
    image_dir_out = output_dir / "images"
    html_dir.mkdir(parents=True, exist_ok=True)
    image_dir_out.mkdir(parents=True, exist_ok=True)
    
    # Lightning 모듈 초기화
    model = TFLOPInferenceLightningModule(config)
    trainer = pl.Trainer(
        accelerator=config.device,
        devices=1,
        precision='32'
    )
    
    # 데이터로더 생성
    dataloader = create_inference_dataloader(
        image_dir=image_dir,
        image_processor=model.image_processor,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 추론 실행
    predictions = trainer.predict(model, dataloaders=dataloader)
    
    # 결과 저장
    for batch_pred, batch in zip(predictions, dataloader):
        for i, (pred, img_path) in enumerate(zip(batch_pred, batch['image_paths'])):
            try:
                # HTML 파일 저장
                html_path = html_dir / f"{img_path.stem}.html"
                save_html(pred['html'], html_path)
                
                # 원본 이미지 복사
                shutil.copy2(img_path, image_dir_out / img_path.name)
                
                print(f"Processed: {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                continue
    
    print(f"\nProcessing complete. Results saved to: {output_dir}")

def main():
    """CLI 인터페이스"""
    from config import InferenceConfig
    config = InferenceConfig()
    
    # Set float32 matmul precision
    torch.set_float32_matmul_precision('high')
    
    process_directory(config)

def construct_table_html_pred(
    otsl_sequence: str,
    bbox_with_text: Dict[int, Dict[str, Union[str, List[float]]]],
    pointer_logits: torch.Tensor,
    confidence_threshold: float = 0.5
) -> str:
    """OTSL과 pointer logits를 사용하여 예측 테이블 생성 (inference 전용)"""
    # 1. OTSL 토큰을 그리드로 변환
    tokens = [t for t in otsl_sequence.split() if t not in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']]
    grid = []
    current_row = []
    
    # 2. 토큰 인덱스와 그리드 위치 매핑
    token_positions = {}  # token_idx -> (row, col)
    token_idx = 0
    current_row_idx = 0
    
    for token in tokens:
        if token == 'NL':
            if current_row:
                current_row.append('NL')
                grid.append(current_row)
                current_row = []
            current_row_idx += 1
            token_positions[token_idx] = (-1, -1)
        else:
            current_row.append(token)
            token_positions[token_idx] = (current_row_idx, len(current_row) - 1)
        token_idx += 1
        
    if current_row:
        grid.append(current_row)

    # 3. 각 셀의 원본 셀 찾기
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

    # 6. pointer_logits를 사용하여 text_cells 매핑
    text_cells = {}  # (row, col) -> text_idx
    pointer_probs = torch.softmax(pointer_logits, dim=1)
    
    # OCR 결과가 있는 경우에만 text mapping 수행
    if bbox_with_text:
        for box_idx in range(pointer_probs.size(0)):
            max_prob, cell_idx = pointer_probs[box_idx].max(dim=0)
            if max_prob.item() >= confidence_threshold and cell_idx < len(token_positions):
                token_idx = cell_idx.item()
                if token_idx in token_positions:
                    row, col = token_positions[token_idx]
                    if row != -1:
                        if (row, col) in text_cells:
                            existing_box_idx = text_cells[(row, col)]
                            existing_prob = pointer_probs[existing_box_idx, cell_idx].item()
                            if max_prob.item() > existing_prob:
                                text_cells[(row, col)] = box_idx
                        else:
                            text_cells[(row, col)] = box_idx

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
                
                # OCR 결과가 있고, 해당 셀에 매핑된 텍스트가 있는 경우에만 텍스트 추가
                if bbox_with_text and (i, j) in text_cells:
                    text_idx = text_cells[(i, j)]
                    if text_idx in bbox_with_text:
                        text = bbox_with_text[text_idx].get('text', '').strip()
                        html.append(f"{cell_tag}{text}</td>")
                    else:
                        html.append(f"{cell_tag}</td>")
                else:
                    html.append(f"{cell_tag}</td>")
                
                for r in range(i, i + rowspan):
                    for c in range(j, j + colspan):
                        processed.add((r, c))
        
        html.append("</tr>")
    
    html.append("</table>")
    return '\n'.join(html)

def save_html(html_content: str, file_path: Path):
    """HTML 파일 저장 및 스타일 추가"""
    styled_html = f"""
    <html>
    <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(styled_html)

if __name__ == '__main__':
    pl.seed_everything(42)
    main()
