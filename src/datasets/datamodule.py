import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Any
from .dataset import TableDataset
from .dataloader import collate_fn

class TableDataModule(pl.LightningDataModule):
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
        
        # 데이터셋 설정 저장
        self.batch_size = train_config.batch_size
        self.num_workers = train_config.num_workers
        self.pin_memory = train_config.pin_memory
        
    def setup(self, stage: Optional[str] = None):
        """데이터셋 초기화"""
        if stage == 'fit' or stage is None:
            self.train_dataset = TableDataset(
                data_dir=self.data_dir,
                split='train',
                max_seq_length=self.model_config.max_seq_length,
                image_size=self.model_config.image_size,
            )
            
            self.val_dataset = TableDataset(
                data_dir=self.data_dir,
                split='val',
                max_seq_length=self.model_config.max_seq_length,
                image_size=self.model_config.image_size,
            )
            
        if stage == 'test':
            self.test_dataset = TableDataset(
                data_dir=self.data_dir,
                split='test',
                max_seq_length=self.model_config.max_seq_length,
                image_size=self.model_config.image_size,
                tokenizer=self.tokenizer
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )