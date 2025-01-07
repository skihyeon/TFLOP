import pytorch_lightning as pl
from typing import Optional, Any
from .dataset import TableDataset, PredictTableDataset
from .dataloader import create_dataloader, create_predict_dataloader
from models.otsl_tokenizer import OTSLTokenizer

class TableDataModule(pl.LightningDataModule):
    """학습/평가용 데이터 모듈"""
    def __init__(
        self,
        data_dir: str,
        model_config: Any,
        train_config: Any,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_config = model_config
        self.train_config = train_config
        
        # 토크나이저는 DataModule에서 한 번만 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=model_config.otsl_max_length
        )
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TableDataset(
                data_dir=self.data_dir,
                split='train',
                image_size=self.model_config.image_size,
                tokenizer=self.tokenizer,
                model_config=self.model_config
            )
            
            self.val_dataset = TableDataset(
                data_dir=self.data_dir,
                split='val',
                image_size=self.model_config.image_size,
                tokenizer=self.tokenizer,
                model_config=self.model_config
            )
    
    def train_dataloader(self):
        return create_dataloader(
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.train_config.batch_size,
            shuffle=False,  # DistributedSampler에서 처리함
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            drop_last=False
        )

    def val_dataloader(self):
        return create_dataloader(
            self.val_dataset,
            self.tokenizer,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            drop_last=False
        )
        
    def test_dataloader(self):
        return create_dataloader(
            self.val_dataset,
            self.tokenizer,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            drop_last=False
        )

class PredictTableDataModule(pl.LightningDataModule):
    """예측 전용 데이터 모듈"""
    def __init__(
        self,
        data_dir: str,
        model_config: Any,
        num_workers: int = 4,
        pin_memory: bool = True,
        ocr_results: Optional[dict] = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_config = model_config
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ocr_results = ocr_results
        
        # 토크나이저 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=model_config.otsl_max_length
        )
        
    def setup(self, stage: Optional[str] = None):
        self.predict_dataset = PredictTableDataset(
            data_dir=self.data_dir,
            image_size=self.model_config.image_size,
            model_config=self.model_config,
            ocr_results=self.ocr_results
        )
    
    def predict_dataloader(self):
        return create_predict_dataloader(
            dataset=self.predict_dataset,
            tokenizer=self.tokenizer,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )