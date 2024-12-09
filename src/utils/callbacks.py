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
            try:
                # loss components 준비
                loss_components = {
                    k.replace('val/', ''): v.item() if torch.is_tensor(v) else v 
                    for k, v in outputs.items() 
                    if k.startswith('val/') and k.endswith('_loss')
                }
                
                # 시각화 저장 (에러 처리를 위해 try 내부로 이동)
                if all(k in outputs for k in ['pred_html', 'true_html', 'pred_otsl', 'true_otsl']):
                    visualize_validation_sample(
                        image=batch['images'][0],
                        boxes=batch['bboxes'][0],
                        pred_html=outputs['pred_html'],
                        true_html=outputs['true_html'],
                        pred_otsl=outputs['pred_otsl'],
                        true_otsl=outputs['true_otsl'],
                        pointer_logits=outputs['pointer_logits'],
                        empty_pointer_logits=outputs['empty_pointer_logits'],
                        step=trainer.current_epoch,  # epoch을 step으로 전달
                        viz_dir=self.viz_dir
                    )
                
                    # HTML 저장
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Table Comparison - Epoch {trainer.current_epoch}</title>
                        <style>
                            .container {{ padding: 20px; }}
                            .metrics {{ margin-bottom: 20px; }}
                            .metric {{ display: inline-block; margin-right: 15px; }}
                            .table-container {{ margin-bottom: 30px; }}
                            table {{ border-collapse: collapse; }}
                            td, th {{ border: 1px solid black; padding: 8px; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h2>Table Comparison - Epoch {trainer.current_epoch}</h2>
                            
                            <div class="metrics">
                                <div class="metric teds">TEDS: {outputs.get('teds', 0.0):.4f}</div>
                                <div class="metric teds">TEDS-Struct: {outputs.get('teds_s', 0.0):.4f}</div>
                                {' '.join(f'<div class="metric loss">{k}: {v:.4f}</div>' for k, v in loss_components.items())}
                            </div>
                            
                            <div class="table-container">
                                <div class="title">Predicted Table</div>
                                {outputs['pred_html']}
                                <div class="pointer-info">
                                    <div>Pointer Confidence: {torch.softmax(outputs['pointer_logits'][0], dim=-1).max().item():.4f}</div>
                                    <div>Empty Pointer Confidence: {torch.sigmoid(outputs['empty_pointer_logits'][0]).max().item():.4f}</div>
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
                    
                    # epoch별로 HTML 파일 저장
                    html_path = self.html_dir / f'epoch_{trainer.current_epoch:04d}.html'
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
            
            except Exception as e:
                print(f"Warning: Validation visualization failed: {str(e)}")