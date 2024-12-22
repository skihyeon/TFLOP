from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset
from models.otsl_tokenizer import OTSLTokenizer
from config import ModelConfig

def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader
    """
    config = ModelConfig()
    layout_prompt_length = config.total_sequence_length - config.otsl_max_length
    batch_size = len(batch)
    
    # 1. 기본 데이터 처리
    image_names = [sample['image_name'] for sample in batch]
    # images = [sample['image'] for sample in batch]
    images = torch.cat([sample['image']['pixel_values'] for sample in batch], dim=0)
    num_boxes = torch.tensor([sample['num_boxes'] for sample in batch])
    
    # 2. OTSL 시퀀스 토큰화
    token_ids_list = torch.stack([
        tokenizer.encode(sample['otsl_tokens_list'], padding=True, return_tensors='pt').squeeze(0)
        for sample in batch
    ])
    
    # 3. bboxes 패딩
    padded_bboxes = torch.zeros(batch_size, layout_prompt_length, 4, dtype=torch.float32)
    for i, (sample, n_box) in enumerate(zip(batch, num_boxes)):
        padded_bboxes[i, :n_box] = torch.tensor(sample['bboxes'][:n_box])
    
    # 4. box indices 처리
    max_mappings = max(
        max(len(mappings) for mappings in sample['box_mappings'].values())
        for sample in batch
    )
    box_indices = torch.full((batch_size, layout_prompt_length, max_mappings), -1)
    
    # sequence_pos 매핑 캐시 생성
    seq_pos_cache = {
        i: {cell['cell_idx']: cell['sequence_pos'] for cell in sample['cells']}
        for i, sample in enumerate(batch)
    }
    
    # box_indices 채우기
    for i, sample in enumerate(batch):
        for cell_idx, bbox_indices in sample['box_mappings'].items():
            if cell_idx < layout_prompt_length:
                sequence_pos = seq_pos_cache[i].get(cell_idx)
                if sequence_pos is not None and 0 <= sequence_pos < config.otsl_max_length:
                    for j, bbox_idx in enumerate(bbox_indices[:max_mappings]):
                        box_indices[i, bbox_idx, j] = sequence_pos
    
    # 5. 마스크 처리
    total_length = layout_prompt_length + config.otsl_max_length
    attention_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    data_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    empty_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        # attention mask 설정
        seq_length = len(sample['otsl_tokens_list'])
        attention_mask[i, :layout_prompt_length + seq_length] = True
        
        # data/empty tag mask 설정
        for cell in sample['cells']:
            pos = cell['sequence_pos']
            if 0 <= pos < config.otsl_max_length:
                mask_pos = layout_prompt_length + pos
                if cell['has_data']:
                    data_tag_mask[i, mask_pos] = True
                else:
                    empty_tag_mask[i, mask_pos] = True

    return {
        'image_names': image_names,
        'images': images,                      # (B, 3, 768, 768)
        'token_ids': token_ids_list,          # (B, otsl_max_length)
        'bboxes': padded_bboxes,              # (B, layout_prompt_length, 4)
        'box_indices': box_indices,           # (B, layout_prompt_length, max_mappings)
        'data_tag_mask': data_tag_mask,       # (B, total_length) - 데이터 있는 C 태그
        'empty_tag_mask': empty_tag_mask,     # (B, total_length) - 데이터 없는 C 태그
        'num_boxes': num_boxes,               # (B)
        'cells': [sample['cells'] for sample in batch],
        'html': [sample['html'] for sample in batch],
        'bbox_with_text': [sample['bbox_with_text'] for sample in batch]
    }
    


def create_dataloader(
    dataset: TableDataset,
    tokenizer: OTSLTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """데이터로더 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=pin_memory,
        drop_last=drop_last
    )

def predict_collate_fn(batch: List[Dict]) -> Dict:
    batch_size = len(batch)
    config = ModelConfig()
    layout_prompt_length = config.total_sequence_length - config.otsl_max_length
    
    # 1. 이미지 처리
    images = torch.stack([item['images'] for item in batch])
    
    # 2. bbox 처리 (layout_prompt_length만큼 패딩)
    bboxes = torch.zeros((batch_size, layout_prompt_length, 4))
    for i, item in enumerate(batch):
        if item['num_boxes'] > 0:
            bboxes[i, :item['num_boxes']] = item['bboxes']
    
    return {
        'image_names': [item['image_name'] for item in batch],
        'images': images,
        'bboxes': bboxes,  # collate_fn에서 패딩
        'bbox_with_text': [item['bbox_with_text'] for item in batch],
        'num_boxes': torch.tensor([item['num_boxes'] for item in batch])
    }

def create_predict_dataloader(
    dataset: TableDataset,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Predict용 DataLoader 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=predict_collate_fn
    )