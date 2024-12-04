from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset
from torch.nn.utils.rnn import pad_sequence
from models.otsl_tokenizer import OTSLTokenizer

def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader
    
    Args:
        batch: List of samples from TableDataset
        tokenizer: OTSLTokenizer instance
    """
    batch_size = len(batch)
    max_boxes = max(sample['num_boxes'] for sample in batch)
    
    # 1. Basic tensors
    images = torch.stack([sample['image'] for sample in batch])
    
    # 2. Tokenize and pad OTSL sequences
    tokens = []
    for sample in batch:
        remaining_length = tokenizer.total_sequence_length - max_boxes
        sample_tokens = tokenizer.encode(
            sample['otsl_str'],
            max_length=remaining_length
        )
        tokens.append(sample_tokens)
    
    max_seq_len = max(len(t) for t in tokens)
    tokens = pad_sequence(
        [torch.tensor(t) for t in tokens],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    # 3. Pad bboxes
    padded_bboxes = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32)
    for i, sample in enumerate(batch):
        num_boxes = sample['num_boxes']
        padded_bboxes[i, :num_boxes] = torch.tensor(sample['bboxes'])
    
    device = padded_bboxes.device  # device 저장
    
    # 4. Create box indices tensor (many-to-one 매핑 지원)
    max_mappings = max(max(len(mappings) for mappings in sample['box_mappings']) 
                      for sample in batch)
    box_indices = torch.full((batch_size, max_boxes, max_mappings), -1, 
                           dtype=torch.long, device=device)
    
    # box_mappings에서 실제 text가 있는 data tag 위치로 매핑
    for i, sample in enumerate(batch):
        num_boxes = sample['num_boxes']
        for box_idx, mappings in enumerate(sample['box_mappings']):
            if mappings:
                # sequence 내 위치를 num_boxes만큼 shift
                seq_indices = [num_boxes + idx for idx in mappings]
                box_indices[i, box_idx, :len(mappings)] = torch.tensor(
                    seq_indices, dtype=torch.long, device=device
                )
    
    # 5. Create attention mask
    attention_mask = torch.zeros(batch_size, tokenizer.total_sequence_length, 
                               dtype=torch.bool, device=device)
    for i, sample in enumerate(batch):
        seq_len = sample['num_boxes'] + len(tokens[i])
        attention_mask[i, :seq_len] = True
    
    # 6. Create data tag mask (text가 있는 C 태그 위치)
    data_tag_mask = torch.zeros(batch_size, tokenizer.total_sequence_length, 
                              dtype=torch.bool, device=device)
    for i, (sample, sample_tokens) in enumerate(zip(batch, tokens)):
        num_boxes = sample['num_boxes']
        has_data = sample['has_data_flags']
        has_data = has_data[:len(sample_tokens)]
        for j, (token_id, has_data_flag) in enumerate(zip(sample_tokens, has_data)):
            # print(f"token_id: {token_id}, has_data_flag: {has_data_flag}")
            if token_id == tokenizer.c_token_id and has_data_flag:
                data_tag_mask[i, num_boxes+j] = True
    
    # print("\nDataloader collate_fn:")
    # for i, sample in enumerate(batch):
        # print(f"\nBatch {i}:")
        # print(f"num_boxes: {sample['num_boxes']}")
        # print(f"box_mappings (sequence positions): {sample['box_mappings']}")
        # print(f"box_indices (real mappings): {box_indices[i]}")
        # true_positions = torch.where(data_tag_mask[i])[0]
        # print(f"True positions in data_tag_mask: {true_positions}")
    
    return {
        'images': images,
        'tokens': tokens,
        'bboxes': padded_bboxes,
        'box_indices': box_indices,
        'attention_mask': attention_mask,
        'data_tag_mask': data_tag_mask,
        'num_boxes': torch.tensor([sample['num_boxes'] for sample in batch]),
        'cells': [sample['cells'] for sample in batch],
        'html': [sample['html'] for sample in batch]
    }

def create_dataloader(
    dataset: TableDataset,
    tokenizer: OTSLTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """데이터로더 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=pin_memory
    ) 