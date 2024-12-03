from typing import Dict, List, Union
import torch
from torch.utils.data import DataLoader
from .dataset import TableDataset

def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    배치 데이터 병합
    
    논문의 모델 입력 요구사항에 맞춰 패딩 처리:
    - 이미지: 768x768 고정 크기 (논문 4.2)
    - 시퀀스: max_seq_length(1376)로 제한 (논문 4.2)
    - boxes: 배치 내 최대 box 수에 맞춰 패딩
    """
    # Basic information
    images = torch.stack([item['image'] for item in batch])
    html = [item['html'] for item in batch]
    cells = [item['cells'] for item in batch]
    
    # Get maximum sizes in batch
    max_boxes = max(item['bboxes'].size(0) for item in batch)
    max_seq_len = min(
        max(item['tokens'].size(0) for item in batch),
        1376  # 논문 4.2
    )
    
    # # 디버깅: 각 샘플의 크기 출력
    # print("\nDebugging size information:")
    # for i, item in enumerate(batch):
    #     print(f"\nSample {i}:")
    #     print(f"bboxes size: {item['bboxes'].size()}")
    #     print(f"row_span_coef size: {item['row_span_coef'].size()}")
    #     print(f"col_span_coef size: {item['col_span_coef'].size()}")
    #     print(f"tokens size: {item['tokens'].size()}")
    # print(f"\nmax_boxes: {max_boxes}")
    # print(f"max_seq_len: {max_seq_len}\n")
    
    # Prepare tensors for batch
    batch_boxes = []
    batch_tokens = []
    batch_attention_mask = []
    batch_data_tag_masks = []
    batch_box_indices = []
    
    for item in batch:
        num_boxes = item['bboxes'].size(0)
        seq_len = min(item['tokens'].size(0), max_seq_len)
        
        # 1. Box coordinates
        padded_boxes = torch.zeros(max_boxes, 4, device=item['bboxes'].device)
        padded_boxes[:num_boxes] = item['bboxes']
        batch_boxes.append(padded_boxes)
        
        # 3. OTSL tokens
        padded_tokens = torch.full((max_seq_len,), 
                                fill_value=1,  # pad_token_id = 1
                                device=item['tokens'].device)
        padded_tokens[:seq_len] = item['tokens'][:seq_len]
        batch_tokens.append(padded_tokens)
        
        attention_mask = torch.zeros(max_seq_len, device=item['tokens'].device)
        attention_mask[:seq_len] = 1
        batch_attention_mask.append(attention_mask)
        
        batch_data_tag_masks.append(item['data_tag_mask'])
        
        # Box indices padding
        curr_mappings = item['box_indices'].size(1)  # 현재 샘플의 매핑 수
        padded_box_indices = torch.full(
            (max_boxes, curr_mappings), 
            -1, 
            device=item['box_indices'].device
        )
        padded_box_indices[:num_boxes] = item['box_indices'][:num_boxes]
        batch_box_indices.append(padded_box_indices)
    
    # Find maximum number of mappings across batch
    max_mappings = max(item['box_indices'].size(1) for item in batch)
    
    # Adjust box_indices padding to match max_mappings
    batch_box_indices_adjusted = []
    for box_indices in batch_box_indices:
        curr_mappings = box_indices.size(1)
        if curr_mappings < max_mappings:
            # Add padding for mappings dimension
            padded = torch.full(
                (box_indices.size(0), max_mappings),
                -1,
                device=box_indices.device
            )
            padded[:, :curr_mappings] = box_indices
            batch_box_indices_adjusted.append(padded)
        else:
            batch_box_indices_adjusted.append(box_indices)

    return {
        'images': images,                                # (B, 3, H, W)
        'bboxes': torch.stack(batch_boxes),              # (B, N, 4)
        'tokens': torch.stack(batch_tokens),             # (B, L)
        'attention_mask': torch.stack(batch_attention_mask), # (B, L)
        'data_tag_mask': torch.stack(batch_data_tag_masks), # (B, L)
        'box_indices': torch.stack(batch_box_indices_adjusted), # (B, N, M)
        'cells': cells,
        'html': html
    }

def create_dataloader(
    dataset: TableDataset,
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
        collate_fn=collate_fn,
        pin_memory=pin_memory
    ) 