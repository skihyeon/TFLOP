from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset
from models.otsl_tokenizer import OTSLTokenizer
from .sampler import OTSLLengthBatchSampler


def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader
    """
    layout_prompt_length = otsl_sequence_length = tokenizer.otsl_sequence_length
    batch_size = len(batch)
    
    image_names = [sample['image_name'] for sample in batch]
    # 1. 기본 텐서들 한번에 처리
    # images = torch.stack([sample['image'] for sample in batch])
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
    
    # 4. box indices - 더 효율적인 처리
    max_mappings = max(max(len(m) for m in sample['box_mappings']) for sample in batch)
    box_indices = torch.full((batch_size, layout_prompt_length, max_mappings), -1)
    
    # 5. attention mask와 data tag mask 통합 처리
    total_length = layout_prompt_length + otsl_sequence_length
    attention_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    data_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    
    # 통합된 루프로 한번에 처리
    for i, (sample, n_box) in enumerate(zip(batch, num_boxes)):
        # Box indices 설정
        for box_idx, mappings in enumerate(sample['box_mappings'][:n_box]):
            if mappings:
                # seq_indices = torch.tensor([layout_prompt_length + idx for idx in mappings])
                seq_indices = torch.tensor(mappings)
                box_indices[i, box_idx, :len(seq_indices)] = seq_indices
        
        # Attention mask 설정
        attention_mask[i, :n_box] = True
        attention_mask[i, layout_prompt_length:][token_ids_list[i] != tokenizer.pad_token_id] = True
        
        # Data tag mask 설정
        data_mask_indices = [j for j, (token_id, has_data) in 
                           enumerate(zip(token_ids_list[i], sample['has_data_flags_list']))
                           if token_id == tokenizer.c_token_id and has_data]
        if data_mask_indices:
            data_tag_mask[i, layout_prompt_length + torch.tensor(data_mask_indices)] = True


    
    batch_dict =  {
        'image_names': image_names,
        'images': images,                     # (B, 3, 768, 768)
        'token_ids': token_ids_list,          # (B, 688)
        'bboxes': padded_bboxes,             # (B, 688, 4)
        'box_indices': box_indices,           # (B, 688, max_mappings)
        'attention_mask': attention_mask,      # (B, 1376)
        'data_tag_mask': data_tag_mask,       # (B, 1376)
        'num_boxes': num_boxes,               # (B)
        'cells': [sample['cells'] for sample in batch],
        'html': [sample['html'] for sample in batch],
        'bbox_with_text': [sample['bbox_with_text'] for sample in batch] # (B, N, 2)
    }
    
    return batch_dict


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