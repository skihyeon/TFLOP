# predict.py
import pytorch_lightning as pl
import torch
from pathlib import Path
import json
from typing import Optional, Dict

from config import ModelConfig, PredictConfig
from models import TFLOPLightningModule
from datasets.datamodule import PredictTableDataModule


def load_model(checkpoint_path: str) -> TFLOPLightningModule:
    """모델 로드 함수"""
    # .pt 파일인 경우
    if checkpoint_path.endswith('.pt'):
        ckpt = torch.load(checkpoint_path)
        # model_config 복원
        model_config = ModelConfig(**ckpt['model_config'])
        # 모델 초기화 및 가중치 로드
        model = TFLOPLightningModule(model_config=model_config)
        model.load_state_dict(ckpt['state_dict'])
    
    # Lightning 체크포인트인 경우
    else:
        ckpt = torch.load(checkpoint_path)
        # Lightning 체크포인트에서 model_config 추출
        if 'hyper_parameters' in ckpt and 'model_config' in ckpt['hyper_parameters']:
            model_config = ModelConfig(**ckpt['hyper_parameters']['model_config'])
            model = TFLOPLightningModule(model_config=model_config)
            # state_dict는 Lightning이 자동으로 로드
            model = TFLOPLightningModule.load_from_checkpoint(
                checkpoint_path,
                model_config=model_config,
                strict=True
            )
        else:
            raise ValueError("Lightning checkpoint does not contain model_config")
    
    return model


def load_ocr_results(gt_path: Path) -> Dict:
    """OCR 결과 로드 함수"""
    ocr_results = {}
    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                filename = data['filename']
                cells = data['html']['cells']
                
                ocr_results[filename] = [
                    {
                        'bbox': cell['bbox'],
                        'text': ''.join(token for token in cell['tokens'] 
                                      if not (token.startswith('<') and token.endswith('>')))
                    }
                    for cell in cells
                    if 'bbox' in cell and 'tokens' in cell
                ]
    return ocr_results


def main():
    # 설정 초기화
    predict_config = PredictConfig()
    
    # 입출력 경로 설정
    input_dir = Path(predict_config.input_dir)
    output_dir = Path(predict_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # OCR 결과 로드 (있는 경우)
    gt_path = input_dir / "gt.txt"
    ocr_results = load_ocr_results(gt_path)
    
    # 모델 로드
    model = load_model(predict_config.checkpoint_path)
    
    # 데이터 모듈 설정
    datamodule = PredictTableDataModule(
        data_dir=str(input_dir),
        model_config=model.model_config,
        num_workers=predict_config.num_workers,
        pin_memory=predict_config.pin_memory,
        ocr_results=ocr_results
    )
    
    # Trainer 설정 (예측용 최소 설정)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[predict_config.gpu_id],
        precision=predict_config.precision,
        logger=False,
        enable_checkpointing=False,
    )
    
    # 예측 실행
    print("Starting predictions...")
    predictions = trainer.predict(model, datamodule=datamodule)
    
    # 결과 저장
    print("\nSaving predictions...")
    for batch_predictions in predictions:
        for pred in batch_predictions:
            image_name = pred['image_name']
            base_name = Path(image_name).stem
            
            # HTML 저장
            html_path = output_dir / f"{base_name}.html"
            with open(html_path, "w", encoding='utf-8') as f:
                f.write(make_html_beautiful(pred['pred_html']))
            
            # OTSL 저장
            otsl_path = output_dir / f"{base_name}.txt"
            with open(otsl_path, "w", encoding='utf-8') as f:
                f.write(pred['pred_otsl'])
            
            print(f"Saved predictions for: {image_name}")
    
    print(f"\nAll predictions saved to {output_dir}")


def make_html_beautiful(html_str: str) -> str:
    """HTML을 가독성 좋게 포맷팅"""
    template = """<!DOCTYPE html>
<html>
<head>
    <title>Table Prediction</title>
    <style>
        .container {{ padding: 20px; }}
        .table-container {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid black; padding: 8px; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
<div class="container">
    <div class="table-container">
        {table}
    </div>
</div>
</body>
</html>"""

    return template.format(table=html_str)


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    main()