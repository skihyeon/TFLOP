# predict.py
import pytorch_lightning as pl
import torch
from pathlib import Path
import json

from config import PredictConfig
from models import TFLOPLightningModule
from datasets.dataset import TableDataset
from datasets.dataloader import create_predict_dataloader


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


def main():    
    config = PredictConfig()
    
    # 1. 이미지 경로 수집
    input_dir = Path(config.input_dir)
    image_paths = list(input_dir.glob("*.[jp][pn][g]"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    # 2. gt.txt에서 OCR 결과 변환
    gt_path = input_dir / "gt.txt"
    ocr_results = {}
    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                filename = data['filename']
                cells = data['html']['cells']
                
                # gt.txt의 cell 정보를 OCR 결과 형식으로 변환
                ocr_results[filename] = [
                    {
                        'bbox': cell['bbox'],
                        'text': ''.join(token for token in cell['tokens'] 
                                      if not (token.startswith('<') and token.endswith('>')))
                    }
                    for cell in cells
                    if 'bbox' in cell and 'tokens' in cell
                ]
    
    # 3. 데이터셋 생성
    predict_dataset = TableDataset(
        data_dir=str(input_dir),  # 현재는 infer_images 디렉토리 사용
        split='val',  # split은 무시됨 (is_predict=True 때문)
        image_size=config.image_size,
        is_predict=True,
        ocr_results=ocr_results
    )
    
    # 4. 데이터로더 생성
    predict_loader = create_predict_dataloader(
        dataset=predict_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # 5. 모델 로드
    model = TFLOPLightningModule.load_from_checkpoint(
        config.checkpoint_path,
        model_config=config,  # PredictConfig를 ModelConfig처럼 사용
        train_config=None,    # inference에서는 불필요
        strict=True
    )
    model.cuda()  # GPU로 이동
    
    # 6. Trainer 설정 - 단순화
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config.gpu_id],
        precision=config.precision,
        logger=False,
        enable_checkpointing=False,
    )
    
    # 7. 예측 실행
    print("Starting predictions...")
    predictions = trainer.predict(model, dataloaders=predict_loader)
    
    # 8. 결과 저장
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving predictions...")
    # predictions는 list of lists 형태 ([batch_predictions])
    for batch_predictions in predictions:
        for pred in batch_predictions:
            image_name = pred['image_name']
            base_name = Path(image_name).stem
            
            # HTML 저장
            with open(output_dir / f"{base_name}.html", "w", encoding='utf-8') as f:
                f.write(make_html_beautiful(pred['pred_html']))
            
            # OTSL 저장
            with open(output_dir / f"{base_name}.txt", "w", encoding='utf-8') as f:
                f.write(pred['pred_otsl'])
            
            print(f"Saved predictions for: {image_name}")
    
    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    main()