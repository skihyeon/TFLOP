import torch
from PIL import Image
from models.inference_module import TFLOPInferenceModule
from config import InferenceConfig
from utils.util import construct_table_html_pred
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
from transformers import AutoImageProcessor

class TableStructureInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Inference 모듈 초기화
        self.model = TFLOPInferenceModule(config)
        self.model.load_from_checkpoint(config.checkpoint_path, self.device)
        
        # 이미지 전처리를 위한 image processor 초기화
        self.image_processor = AutoImageProcessor.from_pretrained(config.swin_model_name)
        self.image_processor.size = (config.image_size, config.image_size)
        
    def prepare_inference_batch(self, image_tensor: torch.Tensor, boxes_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference를 위한 batch 데이터 준비"""
        B = image_tensor.size(0)  # batch size (항상 1)
        layout_prompt_length = self.config.total_sequence_length - self.config.otsl_max_length
        
        # 1. attention mask 생성 (layout prompt 부분은 모두 True)
        attention_mask = torch.ones(
            (B, self.config.total_sequence_length),
            dtype=torch.bool,
            device=self.device
        )
        
        # 2. data tag mask와 empty tag mask 초기화
        data_tag_mask = torch.zeros(
            (B, self.config.total_sequence_length),
            dtype=torch.bool,
            device=self.device
        )
        empty_tag_mask = torch.zeros_like(data_tag_mask)
        
        # layout prompt 부분의 길이만큼만 True로 설정
        if boxes_tensor.size(1) > 0:  # boxes가 있는 경우
            data_tag_mask[:, :boxes_tensor.size(1)] = True
        
        return {
            'images': image_tensor,
            'bboxes': boxes_tensor,
            'data_tag_mask': data_tag_mask,
            'empty_tag_mask': empty_tag_mask,
            'attention_mask': attention_mask
        }
    
    def construct_inference_html(
        self,
        otsl_sequence: str,
        boxes: Optional[torch.Tensor] = None,
        pointer_logits: Optional[torch.Tensor] = None
    ) -> str:
        """Inference 전용 HTML 테이블 생성 함수"""
        # OTSL 토큰을 그리드로 변환
        tokens = [t for t in otsl_sequence.split() if t not in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']]
        grid = []
        current_row = []
        
        for token in tokens:
            if token == 'NL':
                if current_row:
                    grid.append(current_row)
                current_row = []
            else:
                current_row.append(token)
        if current_row:
            grid.append(current_row)
            
        # HTML 생성 시작
        html = ["<table>"]
        
        # 현재 처리 중인 셀의 위치 추적
        processed = set()
        
        for i, row in enumerate(grid):
            html.append("<tr>")
            for j, token in enumerate(row):
                if (i, j) in processed:
                    continue
                    
                if token == 'C':
                    # rowspan과 colspan 계산
                    rowspan = 1
                    colspan = 1
                    
                    # 아래쪽으로 확장 (U 또는 X)
                    for r in range(i + 1, len(grid)):
                        if j < len(grid[r]) and grid[r][j] in ['U', 'X']:
                            rowspan += 1
                        else:
                            break
                            
                    # 오른쪽으로 확장 (L 또는 X)
                    for c in range(j + 1, len(row)):
                        if row[c] in ['L', 'X']:
                            colspan += 1
                        else:
                            break
                    
                    # 셀 태그 시작
                    cell_tag = "<td"
                    if rowspan > 1:
                        cell_tag += f" rowspan='{rowspan}'"
                    if colspan > 1:
                        cell_tag += f" colspan='{colspan}'"
                    cell_tag += ">"
                    
                    # OCR 결과가 있는 경우 텍스트 추가
                    cell_text = ""
                    if boxes is not None and pointer_logits is not None:
                        # pointer_logits에서 현재 셀에 대한 확률 계산
                        probs = torch.softmax(pointer_logits, dim=0)
                        max_prob, box_idx = probs.max(dim=0)
                        
                        if max_prob > 0.5:  # confidence threshold
                            box = boxes[box_idx]
                            # 여기에 box에 해당하는 텍스트를 추가할 수 있음
                            # cell_text = box_texts[box_idx] # OCR 결과가 있다면
                    
                    html.append(f"{cell_tag}{cell_text}</td>")
                    
                    # 처리된 셀들 표시
                    for r in range(i, i + rowspan):
                        for c in range(j, j + colspan):
                            processed.add((r, c))
                            
            html.append("</tr>")
        
        html.append("</table>")
        return '\n'.join(html)
    
    def __call__(self, image_path: str, boxes: Optional[List[List[float]]] = None) -> Dict[str, str]:
        """
        Args:
            image_path: 이미지 경로
            boxes: 텍스트 영역의 bounding box 좌표 리스트 [[x1,y1,x2,y2], ...], 없으면 None
        Returns:
            Dict containing:
                - html: 예측된 HTML 테이블 문자열
                - otsl: 예측된 OTSL 시퀀스
        """
        # 이미지 로드 및 전처리
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
            
        W, H = image.size
        
        # image processor를 사용한 전처리
        image_dict = self.image_processor(image, return_tensors="pt")
        image_tensor = image_dict.pixel_values.to(self.device)
        
        # boxes 처리
        if boxes is None:
            boxes_tensor = torch.zeros((1, 0, 4), dtype=torch.float32, device=self.device)
        else:
            norm_boxes = [[x1/W, y1/H, x2/W, y2/H] for x1,y1,x2,y2 in boxes]
            boxes_tensor = torch.tensor(norm_boxes, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # inference batch 준비
        batch = self.prepare_inference_batch(image_tensor, boxes_tensor)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(batch)
        
        # OTSL 시퀀스 디코딩
        pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
        pred_otsl = self.model.model.tokenizer.decode(pred_tokens)  # model.model로 접근
        
        # HTML 생성 (construct_table_html_pred 대신 새로운 함수 사용)
        pred_html = self.construct_inference_html(
            pred_otsl,
            boxes_tensor[0] if boxes is not None else None,
            outputs['pointer_logits'][0] if boxes is not None else None
        )
        
        return {
            'html': pred_html,
            'otsl': pred_otsl
        }

def main():
    """사용 예시"""
    config = InferenceConfig()
    
    # 추론기 초기화
    inferencer = TableStructureInference(config)
    
    # 입력/출력 디렉토리 생성
    input_dir = Path(config.image_path)
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 목록 (jpg, png 모두 지원)
    image_files = [f for f in input_dir.glob("*.[jp][pn][g]") if f.is_file()]
    
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        
        try:
            # OCR 좌표 없이 추론 실행
            results = inferencer(str(img_path))
            
            # 결과 저장
            output_base = output_dir / img_path.stem
            
            # HTML 저장
            with open(f"{output_base}.html", "w", encoding="utf-8") as f:
                f.write(results['html'])
            
            # OTSL 시퀀스 저장
            with open(f"{output_base}.otsl", "w", encoding="utf-8") as f:
                f.write(results['otsl'])
                
            print(f"Results saved to {output_base}.*")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    main() 