import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Any
from .dataset import TableDataset
from .dataloader import collate_fn, create_dataloader
from models.otsl_tokenizer import OTSLTokenizer

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
        
        # 토크나이저 직접 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=model_config.total_sequence_length // 2
        )
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TableDataset(
                data_dir=self.data_dir,
                split='train',
                image_size=self.model_config.image_size,
                tokenizer=self.tokenizer
            )
            
            self.val_dataset = TableDataset(
                data_dir=self.data_dir,
                split='val',
                image_size=self.model_config.image_size,
                tokenizer=self.tokenizer
            )
    
    def train_dataloader(self):
        return create_dataloader(
            self.train_dataset,
            self.tokenizer,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory
        )
    
    def val_dataloader(self):
        return create_dataloader(
            self.val_dataset,
            self.tokenizer,
            batch_size=self.train_config.batch_size,
            shuffle=False,  # validation은 shuffle 하지 않음
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory
        )