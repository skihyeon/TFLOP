from pytorch_lightning.callbacks import Callback
from .visualize import visualize_validation_sample
import pytorch_lightning as pl
from typing import Dict, Union
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
        """첫 번째 배치에 대해 시각화 수행"""
        if 'pred_html' in outputs:
            try:
                visualize_validation_sample(
                    image_name=outputs['image_name'],
                    image=batch['images'][0],
                    boxes=batch['bboxes'][0],
                    pred_html=outputs['pred_html'],
                    true_html=outputs['true_html'],
                    pred_otsl=outputs['pred_otsl'],
                    true_otsl=outputs['true_otsl'],
                    pointer_logits=outputs['pointer_logits'],
                    empty_pointer_logits=outputs['empty_pointer_logits'],
                    step=trainer.current_epoch,
                    viz_dir=self.viz_dir
                )

                # HTML 파일 저장 (배치 인덱스 포함)
                html_content = self._create_html_content(trainer, outputs)
                html_path = self.html_dir / f'epoch_{trainer.current_epoch:04d}_batch_{batch_idx:04d}.html'
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # 로그 텍스트 저장 (배치 인덱스 포함)
                if 'log_text' in outputs:
                    log_path = self.viz_dir / "logs" / f'epoch_{trainer.current_epoch:04d}_batch_{batch_idx:04d}.txt'
                    log_path.parent.mkdir(exist_ok=True)
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write(outputs['log_text'])
                        
            except Exception as e:
                print(f"Warning: Validation visualization failed for batch {batch_idx}: {str(e)}")

    def _create_html_content(self, trainer, outputs):
        """HTML 컨텐츠 생성"""
        # TEDS 값이 None인 경우 기본값 사용
        teds = outputs.get('teds', None)
        teds_struct = outputs.get('teds_struct', None)
    
        # TEDS 메트릭 HTML 부분
        metrics_html = ""
        if teds is not None and teds_struct is not None:
            metrics_html = f"""
                <div class="metrics">
                    <div class="metric teds">TEDS: {teds:.4f}</div>
                    <div class="metric teds">TEDS-Struct: {teds_struct:.4f}</div>
                </div>
            """
        
        return f"""
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
                
                {metrics_html}
                
                <div class="table-container">
                    <div class="title">Predicted Table</div>
                    {outputs['pred_html']}
                </div>
                
                <div class="table-container">
                    <div class="title">Ground Truth Table</div>
                    {outputs['true_html']}
                </div>
            </div>
        </body>
        </html>
        """

class BestModelSaveCallback(Callback):
    def __init__(self, save_dir: Union[str, Path]):
        super().__init__()
        self.best_teds = 0.0
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_validation_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        if trainer.sanity_checking:
            return
            
        # 수정된 메트릭 이름 사용
        current_teds = trainer.callback_metrics['teds'].item()
        
        if current_teds > self.best_teds:
            self.best_teds = current_teds
            save_path = self.save_dir / f"{pl_module.train_config.exp_name}_best.pt"
            
            try:
                torch.save({
                    'state_dict': pl_module.state_dict(),
                    'model_config': pl_module.model_config.__dict__,
                    'train_config': pl_module.train_config.__dict__,
                    'teds_score': current_teds
                }, save_path)
                print(f"Saved best model with TEDS score: {current_teds:.4f}")
            except Exception as e:
                print(f"Error saving best model: {str(e)}")