from pytorch_lightning.callbacks import Callback
from .visualize import visualize_validation_sample
import pytorch_lightning as pl
from typing import Dict
import torch
from pathlib import Path
import json

class ValidationVisualizationCallback(Callback):
    def __init__(self, viz_dir: str):
        super().__init__()
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML 저장을 위한 디렉토리 생성
        self.html_dir = self.viz_dir / "html"
        self.html_dir.mkdir(exist_ok=True)
                
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        if batch_idx == 0:  # 첫 번째 배치만 시각화
            # val/ 접두사 제거한 loss components 준비
            loss_components = {
                k.replace('val/', ''): v.item() if torch.is_tensor(v) else v 
                for k, v in outputs.items() 
                if k in ['val/loss', 'val/cls_loss', 'val/ptr_loss', 'val/empty_ptr_loss']
            }
            
            # 시각화 저장
            visualize_validation_sample(
                image=batch['images'][0],
                boxes=batch['bboxes'][0],
                pred_html=outputs['pred_html'],
                true_html=outputs['true_html'],
                pred_otsl=outputs['pred_otsl'],
                true_otsl=outputs['true_otsl'],
                pointer_logits=outputs.get('pointer_logits', None),
                step=trainer.global_step,
                loss_components=loss_components,
                viz_dir=self.viz_dir
            )
            
            # HTML 저장
            if 'pred_html' in outputs and 'true_html' in outputs:
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Table Comparison - Step {trainer.global_step}</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 20px auto;
                            max-width: 1200px;
                            padding: 20px;
                        }}
                        .container {{
                            display: flex;
                            flex-direction: column;
                            gap: 20px;
                        }}
                        .table-container {{
                            border: 1px solid #ccc;
                            padding: 20px;
                            border-radius: 5px;
                        }}
                        .metrics {{
                            display: flex;
                            gap: 10px;
                            margin-bottom: 20px;
                        }}
                        .metric {{
                            background: #f0f0f0;
                            padding: 10px;
                            border-radius: 4px;
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
                        .pointer-info {{
                            margin-top: 10px;
                            font-size: 0.9em;
                            color: #666;
                        }}
                        .confidence {{
                            color: #007bff;
                            font-weight: bold;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>Table Comparison - Step {trainer.global_step}</h2>
                        
                        <div class="metrics">
                            <div class="metric">TEDS: {outputs['teds']:.4f}</div>
                            <div class="metric">TEDS-Struct: {outputs['teds_s']:.4f}</div>
                            {' '.join(f'<div class="metric">{k}: {v:.4f}</div>' for k, v in loss_components.items())}
                        </div>
                        
                        <div class="table-container">
                            <div class="title">Predicted Table</div>
                            {outputs['pred_html']}
                            <div class="pointer-info">
                                Pointer Confidence: {torch.softmax(outputs['pointer_logits'][0], dim=-1).max().item():.4f}
                            </div>
                        </div>
                        
                        <div class="table-container">
                            <div class="title">Ground Truth Table</div>
                            {outputs['true_html']}
                        </div>
                    </div>
                </body>
                </html>
                """
                
                # 스텝별로 HTML 파일 저장
                html_path = self.html_dir / f'step_{trainer.global_step:08d}.html'
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)