import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image
import torchvision.transforms as transforms

from models.module import TFLOPLightningModule
from config import ModelConfig, InferenceConfig
from utils.util import construct_table_html
from models.tflop import TFLOP


class TFLOPInference(TFLOPLightningModule):
    def __init__(
        self, 
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        checkpoint_path: Optional[str] = None
    ):
        super().__init__(model_config=model_config, train_config=None)
        
        # 추론 모드로 모델 초기화
        self.model = TFLOP(model_config, inference_mode=True)
        
        # 체크포인트 로드
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict)
        
        # 디바이스 설정
        self.device = torch.device(inference_config.device)
        self.to(self.device)
        self.eval()
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((model_config.image_size, model_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """이미지 전처리"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    def preprocess_text_regions(
        self,
        text_regions: List[Dict[str, Union[str, List[float]]]],
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Union[str, List[float]]]]:
        """Text region bbox 좌표 정규화"""
        normalized_regions = []
        for region in text_regions:
            x0, y0, x1, y1 = region['bbox']
            normalized_bbox = [
                x0 / image_width,
                y0 / image_height,
                x1 / image_width,
                y1 / image_height
            ]
            normalized_regions.append({
                'text': region['text'],
                'bbox': normalized_bbox
            })
        return normalized_regions
    
    @torch.no_grad()
    def inference(
        self,
        image_path: Union[str, Path],
        text_regions: List[Dict[str, Union[str, List[float]]]],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        테이블 이미지 추론
        
        Args:
            image_path: 이미지 경로
            text_regions: text region 정보 리스트 [{'text': str, 'bbox': [x0, y0, x1, y1]}, ...]
            image_width: 원본 이미지 너비 (bbox 정규화에 사용)
            image_height: 원본 이미지 높이 (bbox 정규화에 사용)
            output_dir: 결과 저장 디렉토리 (지정 시 HTML 파일 저장)
            
        Returns:
            Dict[str, str]: 'html': HTML 테이블, 'otsl': OTSL 시퀀스
        """
        self.eval()
        
        # 1. 이미지 전처리
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # 2. Text region 전처리
        if image_width is None or image_height is None:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        
        normalized_regions = self.preprocess_text_regions(
            text_regions, image_width, image_height
        )
        
        # bbox 텐서 생성
        bboxes = torch.tensor(
            [region['bbox'] for region in normalized_regions],
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # batch dimension 추가
        
        # 3. 모델 추론
        outputs = self.model(
            images=image,
            text_regions=bboxes,
            labels=None,
            attention_mask=None,
            row_spans=None,
            col_spans=None
        )
        
        # 4. OTSL 시퀀스 디코딩
        pred_otsl = self.model.tokenizer.decode(
            outputs['tag_logits'][0].argmax(dim=-1).cpu().tolist()
        )
        
        # 5. HTML 테이블 생성
        html_table = construct_table_html(
            pred_otsl,
            normalized_regions,
            outputs['pointer_logits'][0]
        )
        
        # 6. 결과 저장 (지정된 경우)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Table Result - {image_name}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px auto;
                        max-width: 1200px;
                        padding: 20px;
                    }}
                    .container {{
                        border: 1px solid #ccc;
                        padding: 20px;
                        border-radius: 5px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 10px 0;
                    }}
                    td, th {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Table Result - {image_name}</h2>
                    {html_table}
                </div>
            </body>
            </html>
            """
            
            with open(output_dir / f"{image_name}.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return {
            'html': html_table,
            'otsl': pred_otsl
        }


# 사용 예시
if __name__ == '__main__':
    # 설정
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    
    # 추론기 초기화
    inferencer = TFLOPInference(
        model_config=model_config,
        inference_config=inference_config,
        checkpoint_path='checkpoints/best_model.pt'
    )
    
    # 샘플 데이터
    image_path = 'samples/table.png'
    text_regions = [
        {'text': 'Sample', 'bbox': [100, 100, 200, 150]},
        {'text': 'Text', 'bbox': [210, 100, 300, 150]},
    ]
    
    # 추론
    result = inferencer.inference(
        image_path=image_path,
        text_regions=text_regions,
        output_dir='results'
    )
    
    print("Generated HTML:")
    print(result['html']) 