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
            otsl_sequence_length=model_config.otsl_max_length
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,  # DistributedSampler에서 처리함
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
            drop_last=False
        )