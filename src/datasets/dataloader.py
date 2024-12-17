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
    max_mappings = max(max(len(m) for m in sample['box_mappings']) for sample in batch)
    box_indices = torch.full((batch_size, layout_prompt_length, max_mappings), -1)
    
    # box_mappings 값을 box_indices에 채우기
    for i, sample in enumerate(batch):
        for box_idx, mappings in enumerate(sample['box_mappings']):
            if box_idx < layout_prompt_length:  # layout prompt 길이 체크
                if mappings:  # 매핑이 있는 경우
                    for j, pos in enumerate(mappings):
                        if j < max_mappings:  # max_mappings 범위 체크
                            # 이미 dataset에서 올바른 position을 반환하므로 추가 조정 불필요
                            if pos < config.otsl_max_length:  # 시퀀스 길이 체크
                                box_indices[i, box_idx, j] = pos
                            else:
                                print(f"Warning: Position {pos} exceeds max length for batch {i}, box {box_idx}")
    
    # 5. attention mask, data tag mask, empty tag mask 처리
    total_length = layout_prompt_length + config.otsl_max_length
    attention_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)
    data_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)  # non-empty C 태그용
    empty_tag_mask = torch.zeros(batch_size, total_length, dtype=torch.bool)  # empty C 태그용
    
    for i, sample in enumerate(batch):
        # 1. attention mask 처리
        # - layout prompt 부분 (bboxes가 있는 부분까지만)
        attention_mask[i, :sample['num_boxes']] = True
        # - OTSL sequence 부분 (실제 토큰이 있는 부분까지만)
        seq_length = len(sample['otsl_tokens_list'])
        attention_mask[i, layout_prompt_length:layout_prompt_length + seq_length] = True
        
        # 2. data tag masks 처리
        for j, (token, has_data) in enumerate(zip(sample['otsl_tokens_list'], sample['has_data_flags_list'])):
            if token == 'C':
                pos = layout_prompt_length + j
                if has_data:
                    # non-empty C 태그
                    data_tag_mask[i, pos] = True
                else:
                    # empty C 태그
                    empty_tag_mask[i, pos] = True

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