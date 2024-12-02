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
    # batch_row_spans = []
    # batch_col_spans = []
    batch_tokens = []
    batch_attention_mask = []
    batch_data_tag_masks = []
    batch_box_indices = []
    batch_empty_masks = []
    
    for item in batch:
        num_boxes = item['bboxes'].size(0)
        seq_len = min(item['tokens'].size(0), max_seq_len)
        
        # 1. Box coordinates
        padded_boxes = torch.zeros(max_boxes, 4, device=item['bboxes'].device)
        padded_boxes[:num_boxes] = item['bboxes']
        batch_boxes.append(padded_boxes)
        
        # # 2. Span coefficients - 실제 coefficient matrix 크기 유지
        # row_coef = item['row_span_coef']  # (N, N)
        # col_coef = item['col_span_coef']  # (N, N)
        
        # # 디버깅: span coefficient 크기 출력
        # print(f"Processing span coefficients:")
        # print(f"row_coef size: {row_coef.size()}")
        # print(f"col_coef size: {col_coef.size()}")
        
        # 원본 크기 그대로 배치에 추가
        # batch_row_spans.append(row_coef)
        # batch_col_spans.append(col_coef)
        
        # 3. OTSL tokens
        padded_tokens = torch.full((max_seq_len,), 
                                 fill_value=1,  # pad_token_id = 1
                                 device=item['tokens'].device)
        padded_tokens[:seq_len] = item['tokens'][:seq_len]
        batch_tokens.append(padded_tokens)
        
        attention_mask = torch.zeros(max_seq_len, device=item['tokens'].device)
        attention_mask[:seq_len] = 1
        batch_attention_mask.append(attention_mask)
        
        # 4. Data tag mask
        padded_data_tag_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=item['data_tag_mask'].device)
        padded_data_tag_mask[:seq_len] = item['data_tag_mask'][:seq_len]
        batch_data_tag_masks.append(padded_data_tag_mask)
        
        # 5. Empty mask
        padded_empty_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=item['empty_mask'].device)
        padded_empty_mask[:seq_len] = item['empty_mask'][:seq_len]
        batch_empty_masks.append(padded_empty_mask)
        
        # 6. Box indices
        padded_box_indices = torch.full((max_boxes,), -1, device=item['box_indices'].device)
        padded_box_indices[:num_boxes] = item['box_indices'][:num_boxes]
        batch_box_indices.append(padded_box_indices)
    
    # 배치 내에서 가장 큰 coefficient matrix 크기 찾기
    # max_coef_size = max(coef.size(0) for coef in batch_row_spans)
    
    # # coefficient matrices를 max_coef_size에 맞춰 패딩
    # padded_row_spans = []
    # padded_col_spans = []
    # for row_coef, col_coef in zip(batch_row_spans, batch_col_spans):
    #     curr_size = row_coef.size(0)
    #     if curr_size < max_coef_size:
    #         # Zero padding
    #         padded_row = torch.zeros(max_coef_size, max_coef_size, device=row_coef.device)
    #         padded_col = torch.zeros(max_coef_size, max_coef_size, device=col_coef.device)
    #         padded_row[:curr_size, :curr_size] = row_coef
    #         padded_col[:curr_size, :curr_size] = col_coef
    #         padded_row_spans.append(padded_row)
    #         padded_col_spans.append(padded_col)
    #     else:
    #         padded_row_spans.append(row_coef)
    #         padded_col_spans.append(col_coef)
    #     print(f"row coef: {row_coef.size()}, col coef: {col_coef.size()}, coef padded: {padded_row.size()}")
        

    return {
        'images': images,                                # (B, 3, H, W)
        'bboxes': torch.stack(batch_boxes),              # (B, N, 4)
        # 'row_span_coef': torch.stack(padded_row_spans),  # (B, M, M) where M is max coef size
        # 'col_span_coef': torch.stack(padded_col_spans),  # (B, M, M)
        'tokens': torch.stack(batch_tokens),             # (B, L)
        'attention_mask': torch.stack(batch_attention_mask), # (B, L)
        'data_tag_mask': torch.stack(batch_data_tag_masks), # (B, L)
        'empty_mask': torch.stack(batch_empty_masks),       # (B, L)
        'box_indices': torch.stack(batch_box_indices),       # (B, N)
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