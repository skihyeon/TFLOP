from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset
from models.otsl_tokenizer import OTSLTokenizer
from .sampler import OTSLLengthBatchSampler
from config import ModelConfig

def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader
    """
    config = ModelConfig()
    layout_prompt_length = config.total_sequence_length - config.otsl_max_length
    batch_size = len(batch)
    
    image_names = [sample['image_name'] for sample in batch]
    # 1. 기본 텐서들 한번에 처리
    images = torch.cat([sample['image']['pixel_values'] for sample in batch], dim=0)
    num_boxes = torch.tensor([sample['num_boxes'] for sample in batch])
    
    # 2. OTSL 시퀀스 토큰화 - 벡터화 처리
    token_ids_list = torch.stack([
        tokenizer.encode(sample['otsl_tokens_list'], padding=True, return_tensors='pt').squeeze(0)
        for sample in batch
    ])
    
    # 3. bboxes 패딩 - 벡터화 처리
    padded_bboxes = torch.zeros(batch_size, layout_prompt_length, 4, dtype=torch.float32)
    for i, (sample, n_box) in enumerate(zip(batch, num_boxes)):
        padded_bboxes[i, :n_box] = torch.tensor(sample['bboxes'][:n_box])
    
    # 4. box indices 처리
    max_mappings = max(
        max(len(mappings) for mappings in sample['box_mappings'].values())
        for sample in batch
    )
    box_indices = torch.full((batch_size, layout_prompt_length, max_mappings), -1)
    
    # box_mappings 값을 box_indices에 채우기
    for i, sample in enumerate(batch):
        for cell_idx, bbox_indices in sample['box_mappings'].items():
            if cell_idx < layout_prompt_length:  # layout prompt 길이 체크
                for j, bbox_idx in enumerate(bbox_indices):
                    if j < max_mappings:  # max_mappings 범위 체크
                        # sequence_pos 찾기
                        for cell in sample['cells']:
                            if cell['cell_idx'] == cell_idx:
                                sequence_pos = cell['sequence_pos']
                                if 0 <= sequence_pos < config.otsl_max_length:
                                    box_indices[i, bbox_idx, j] = sequence_pos
    
    # 5. attention mask, data tag mask, empty tag mask 처리
    total_length = layout_prompt_length + config.otsl_max_length
    attention_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    data_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    empty_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        # 실제 토큰이 있는 위치까지만 attention mask 설정
        seq_length = len(sample['otsl_tokens_list'])
        attention_mask[i, :layout_prompt_length + seq_length] = True
        
        # data tag mask와 empty tag mask 설정
        for cell in sample['cells']:
            pos = cell['sequence_pos']
            if 0 <= pos < config.otsl_max_length:
                if cell['has_data']:
                    # 데이터가 있는 셀
                    data_tag_mask[i, layout_prompt_length + pos] = True
                else:
                    # 빈 셀 (데이터가 없는 셀)
                    empty_tag_mask[i, layout_prompt_length + pos] = True

    return {
        'image_names': image_names,
        'images': images,                     # (B, 3, 768, 768)
        'token_ids': token_ids_list,          # (B, otsl_max_length)
        'bboxes': padded_bboxes,             # (B, layout_prompt_length, 4)
        'box_indices': box_indices,           # (B, layout_prompt_length, max_mappings)
        'attention_mask': attention_mask,      # (B, total_length)
        'data_tag_mask': data_tag_mask,       # (B, total_length) - non-empty C 태그만 True
        'empty_tag_mask': empty_tag_mask,     # (B, total_length) - empty C 태그만 True
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
    use_length_sampler: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """데이터로더 생성"""
    if use_length_sampler:
        batch_sampler = OTSLLengthBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
            pin_memory=pin_memory
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
            pin_memory=pin_memory,
            drop_last=drop_last
        )