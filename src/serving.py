import gradio as gr
import torch
from pathlib import Path
from PIL import Image
import tempfile
from config import InferenceConfig
from inference import TFLOPInferenceLightningModule, save_html, construct_table_html_pred
from typing import Dict, Union, List, Tuple

class TFLOPGradioInterface:
    def __init__(self, config_path: str = None):
        # Config 초기화
        self.config = InferenceConfig()
        if config_path:
            # config 파일에서 설정 로드 (필요한 경우)
            pass
            
        # TFLOP 모델 초기화
        self.model = TFLOPInferenceLightningModule(self.config)
        self.model.setup()
        
        # 모델을 GPU로 이동
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # GPU 설정
        torch.set_float32_matmul_precision('high')
        
    def predict(self, image) -> Tuple[str, str]:
        """이미지를 받아서 테이블 HTML 반환"""
        try:
            # 임시 파일로 이미지 저장
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                if isinstance(image, str):
                    # 이미 파일 경로인 경우
                    image_path = image
                else:
                    # Gradio에서 전달된 이미지 객체인 경우
                    image.save(tmp_file.name)
                    image_path = tmp_file.name
            
            # 데이터 준비
            batch = {
                'images': self.model.image_processor(
                    Image.open(image_path).convert('RGB'),
                    return_tensors="pt"
                )['pixel_values'].to(self.device),
                'bboxes': torch.zeros(
                    1, 
                    self.config.total_sequence_length // 2, 
                    4, 
                    device=self.device
                ),
                'token_ids': torch.zeros(
                    1,
                    self.config.total_sequence_length,
                    dtype=torch.long,
                    device=self.device
                ),
                'attention_mask': torch.ones(
                    1,
                    self.config.total_sequence_length,
                    dtype=torch.bool,
                    device=self.device
                )
            }
            
            # 추론
            with torch.no_grad():
                outputs = self.model.model(batch)
                
                # 결과 처리
                pred_tokens = outputs['tag_logits'][0].argmax(dim=-1)
                pred_otsl = self.model.tokenizer.decode(pred_tokens.cpu().tolist())
                
                # HTML 생성
                html = construct_table_html_pred(
                    pred_otsl,
                    {},  # OCR 결과 없음
                    outputs['pointer_logits'][0]
                )
                
                # HTML 스타일 추가
                styled_html = f"""
                <div style="border:1px solid #ccc; padding:10px; overflow:auto; max-height:600px;">
                    <style>
                        table {{
                            border-collapse: collapse;
                            width: 100%;
                            margin: 20px 0;
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
                    {html}
                </div>
                """
                
                return styled_html, pred_otsl
        except Exception as e:
            return f"<div style='color:red;'>Error processing image: {str(e)}</div>", ""

def create_interface():
    # 인터페이스 초기화
    interface = TFLOPGradioInterface()
    
    # Gradio UI 생성
    demo = gr.Interface(
        fn=interface.predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.HTML(label="Table Preview"),
            gr.Textbox(label="OTSL Sequence")
        ],
        title="TFLOP Table Structure Recognition",
        description="Upload an image containing a table to recognize its structure.",
        examples=[
            # 예시 이미지 경로들
            ["infer_images/PMC1552090_006_00.png"],
            ["infer_images/PMC543452_002_00.png"]
        ]
    )
    
    return demo

if __name__ == "__main__":
    from setproctitle import setproctitle
    setproctitle("Gradio")
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", share=True) 