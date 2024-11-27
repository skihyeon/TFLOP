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
    
    # 모든 시퀀스를 동일한 길이로 맞춤
    for item in batch:
        item['tokens'] = item['tokens'][:max_seq_len]
    
    # Prepare tensors for batch
    batch_boxes = []      # box coordinates
    batch_row_spans = []  # row span matrices
    batch_col_spans = []  # column span matrices
    batch_tokens = []     # OTSL tokens
    batch_attention_mask = [] # attention mask for tokens
    
    for item in batch:
        num_boxes = item['bboxes'].size(0)
        seq_len = min(item['tokens'].size(0), max_seq_len)
        
        # 1. Box coordinates (normalized)
        padded_boxes = torch.zeros(max_boxes, 4, device=item['bboxes'].device)
        padded_boxes[:num_boxes] = item['bboxes']
        batch_boxes.append(padded_boxes)
        
        # 2. Span matrices
        # row_spans와 col_spans는 이제 (max_rows, max_cols) 형태
        row_span_shape = item['row_spans'].shape
        col_span_shape = item['col_spans'].shape
        
        padded_row_spans = torch.zeros(max_boxes, max_boxes, device=item['row_spans'].device)
        padded_row_spans[:row_span_shape[0], :row_span_shape[1]] = item['row_spans']
        batch_row_spans.append(padded_row_spans)
        
        padded_col_spans = torch.zeros(max_boxes, max_boxes, device=item['col_spans'].device)
        padded_col_spans[:col_span_shape[0], :col_span_shape[1]] = item['col_spans']
        batch_col_spans.append(padded_col_spans)
        
        # 3. OTSL tokens
        padded_tokens = torch.full((max_seq_len,), 
                                 fill_value=item['tokens'].new_zeros(1)[0],  # padding token id
                                 device=item['tokens'].device)
        padded_tokens[:seq_len] = item['tokens'][:seq_len]
        batch_tokens.append(padded_tokens)
        
        # 4. Attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(max_seq_len, device=item['tokens'].device)
        attention_mask[:seq_len] = 1
        batch_attention_mask.append(attention_mask)
    
    # Stack all tensors
    return {
        'images': images,                                # (B, 3, 768, 768)
        'bboxes': torch.stack(batch_boxes),              # (B, N, 4)
        'row_spans': torch.stack(batch_row_spans),       # (B, N, N)
        'col_spans': torch.stack(batch_col_spans),       # (B, N, N)
        'tokens': torch.stack(batch_tokens),             # (B, L)
        'attention_mask': torch.stack(batch_attention_mask), # (B, L)
        'cells': cells,                                  # List[List[Dict]] - cell 정보
        'html': html                                     # List[Dict] - 원본 HTML 구조
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