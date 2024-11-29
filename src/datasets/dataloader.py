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
    
    # Prepare tensors for batch
    batch_boxes = []      # box coordinates
    batch_row_spans = []  # row span matrices
    batch_col_spans = []  # column span matrices
    batch_tokens = []     # OTSL tokens
    batch_attention_mask = [] # attention mask for tokens
    batch_data_tag_masks = []  # data tag masks
    batch_box_indices = []  # box indices for pointer loss
    
    for item in batch:
        num_boxes = item['bboxes'].size(0)
        seq_len = min(item['tokens'].size(0), max_seq_len)
        
        # 1. Box coordinates
        padded_boxes = torch.zeros(max_boxes, 4, device=item['bboxes'].device)
        padded_boxes[:num_boxes] = item['bboxes']
        batch_boxes.append(padded_boxes)
        
        # 2. Span matrices - 실제 box 개수만큼만 패딩
        row_spans = item['row_spans']  # (N, N)
        col_spans = item['col_spans']  # (N, N)
        
        # max_boxes 크기로 한 번만 패딩
        padded_row_spans = torch.zeros(max_boxes, max_boxes, device=row_spans.device)
        padded_col_spans = torch.zeros(max_boxes, max_boxes, device=col_spans.device)
        
        # 실제 span 정보만 복사
        padded_row_spans[:num_boxes, :num_boxes] = row_spans[:num_boxes, :num_boxes]
        padded_col_spans[:num_boxes, :num_boxes] = col_spans[:num_boxes, :num_boxes]
        
        batch_row_spans.append(padded_row_spans)
        batch_col_spans.append(padded_col_spans)
        
        # 3. OTSL tokens
        padded_tokens = torch.full((max_seq_len,), 
                                 fill_value=1,  # pad_token_id = 1
                                 device=item['tokens'].device)
        padded_tokens[:seq_len] = item['tokens'][:seq_len]
        batch_tokens.append(padded_tokens)
        
        attention_mask = torch.zeros(max_seq_len, device=item['tokens'].device)
        attention_mask[:seq_len] = 1
        batch_attention_mask.append(attention_mask)
        
        # 4. Data tag mask - 시퀀스 길이까지만
        padded_data_tag_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=item['data_tag_mask'].device)
        padded_data_tag_mask[:seq_len] = item['data_tag_mask'][:seq_len]
        batch_data_tag_masks.append(padded_data_tag_mask)
        
        # 5. Box indices - 실제 box 개수만큼만
        batch_box_indices.append(item['box_indices'][:num_boxes])
    
    return {
        'images': images,                                # (B, 3, 768, 768)
        'bboxes': torch.stack(batch_boxes),              # (B, N, 4)
        'row_spans': torch.stack(batch_row_spans),       # (B, N, N)
        'col_spans': torch.stack(batch_col_spans),       # (B, N, N)
        'tokens': torch.stack(batch_tokens),             # (B, L)
        'attention_mask': torch.stack(batch_attention_mask), # (B, L)
        'data_tag_mask': torch.stack(batch_data_tag_masks), # (B, L)
        'box_indices': torch.nn.utils.rnn.pad_sequence(  # (B, N')
            batch_box_indices, 
            batch_first=True,
            padding_value=-1
        ),
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