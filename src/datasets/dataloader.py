from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset
from models.otsl_tokenizer import OTSLTokenizer



def collate_fn(batch, tokenizer):
    """
    Collate function for DataLoader
    """
    layout_prompt_length = otsl_sequence_length = tokenizer.otsl_sequence_length
    batch_size = len(batch)
    
    # 1. 기본 텐서들 한번에 처리
    images = torch.stack([sample['image'] for sample in batch])
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
        'images': images,                     # (B, 3, 768, 768)
        'token_ids': token_ids_list,          # (B, 688)
        'bboxes': padded_bboxes,             # (B, 688, 4)
        'box_indices': box_indices,           # (B, 688, max_mappings)
        'attention_mask': attention_mask,      # (B, 1376)
        'data_tag_mask': data_tag_mask,       # (B, 1376)
        'num_boxes': num_boxes,               # (B)
        'cells': [sample['cells'] for sample in batch],
        'html': [sample['html'] for sample in batch]
    }
    
    return batch_dict
# def collate_fn(batch, tokenizer):
#     """
#     Collate function for DataLoader
    
#     Args:
#         batch: List of samples from TableDataset
#         tokenizer: OTSLTokenizer instance
#     """
#     layout_prompt_length = otsl_sequence_length = tokenizer.otsl_sequence_length
#     batch_size = len(batch)
    
#     # 1. Basic tensors
#     images = torch.stack([sample['image'] for sample in batch])
    
#     # 2. Tokenize and pad OTSL sequences
#     token_ids_list = torch.zeros((batch_size, otsl_sequence_length), dtype=torch.long)
#     for i, sample in enumerate(batch):
#         sample_token_ids = tokenizer.encode(
#             sample['otsl_tokens_list'],
#             padding=True,
#             return_tensors='pt'
#         )
#         token_ids_list[i] = sample_token_ids.squeeze(0)  # (1, L) -> (L)

#     # 3. Pad bboxes (1376 // 2 = 688)
#     padded_bboxes = torch.zeros(batch_size, layout_prompt_length, 4, dtype=torch.float32)
#     for i, sample in enumerate(batch):
#         num_boxes = sample['num_boxes']       # 최대 688개
#         padded_bboxes[i, :num_boxes] = torch.tensor(sample['bboxes'][:num_boxes]) 
    
#     # 4. Box indices
#     ## 점검 필요
#     max_mappings = max(max(len(mappings) for mappings in sample['box_mappings']) 
#                       for sample in batch)                                          # batch 내 최대 mapping 개수
#     box_indices = torch.full((batch_size, layout_prompt_length, max_mappings), -1)  # mapping 되지 않으면 -1    
    
#     for i, sample in enumerate(batch):
#         num_boxes = sample['num_boxes']
#         for box_idx, mappings in enumerate(sample['box_mappings'][:num_boxes]):
#             if mappings:
#                 seq_indices = [layout_prompt_length + idx for idx in mappings]
#                 box_indices[i, box_idx, :len(seq_indices)] = torch.tensor(seq_indices)      # (B, 688, max_mappings)
    
#     # 5. Attention & data tag masks
#     attention_mask = torch.zeros(batch_size, layout_prompt_length + otsl_sequence_length, dtype=torch.bool)
#     data_tag_mask = torch.zeros(batch_size, layout_prompt_length + otsl_sequence_length, dtype=torch.bool)
    
#     for i, sample in enumerate(batch):
#         num_boxes = sample['num_boxes']
#         attention_mask[i, :num_boxes] = True
        
#         for j, token_id in enumerate(token_ids_list[i]):
#             if token_id != tokenizer.pad_token_id:
#                 attention_mask[i, layout_prompt_length + j] = True
        
#         # Data tag mask 채우기
#         has_data = sample['has_data_flags_list']
#         for j, (token_id, has_data_flag) in enumerate(zip(token_ids_list[i], has_data)):
#             if token_id == tokenizer.c_token_id and has_data_flag:
#                 data_tag_mask[i, layout_prompt_length + j] = True
    
#     return {
#         'images': images,                            # (B, 3, 1024, 1024)
#         'token_ids': token_ids_list,                   # (B, 688)
#         'bboxes': padded_bboxes,                    # (B, 688, 4)
#         'box_indices': box_indices,                   # (B, 688, )
#         'attention_mask': attention_mask,              # (B, 1376)
#         'data_tag_mask': data_tag_mask,                # (B, 1376)
#         'num_boxes': torch.tensor([sample['num_boxes'] for sample in batch]),  # (N)
#         'cells': [sample['cells'] for sample in batch],
#         'html': [sample['html'] for sample in batch]
#     }

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