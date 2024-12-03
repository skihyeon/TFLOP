from typing import Dict, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import SwinModel, BartForConditionalGeneration, BartConfig
from transformers import Swinv2Model
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
import torch.nn.functional as F
from utils.util import get_coef_matrix

class TFLOP(nn.Module):
    """
    TFLOP: Table Structure Recognition Framework with Layout Pointer Mechanism
    
    논문 Section 3.1 Overall Architecture 참조:
    - 4개의 주요 모듈: image encoder, layout encoder, logical structure decoder, layout pointer
    """
    def __init__(self, config: Any, inference_mode: bool = False) -> None:
        super().__init__()
        
        self.config = config
        
        # Tokenizer 초기화
        self.tokenizer = OTSLTokenizer(
            vocab_size=config.vocab_size,
            max_length=config.max_seq_length
        )
        

        # 학습 모드: pretrained 모델 로드
        self.image_encoder = SwinModel.from_pretrained(config.swin_model_name)
        # self.image_encoder = Swinv2Model.from_pretrained(config.swin_model_name)
        # BART 설정 및 초기화
        bart_config = BartConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_seq_length + 2,
            d_model=config.feature_dim,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            encoder_attention_heads=config.encoder_attention_heads,
            decoder_attention_heads=config.decoder_attention_heads,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            is_encoder_decoder=True,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            forced_eos_token_id=self.tokenizer.eos_token_id,
            scale_embedding=True,
            use_cache=True,
        )
        self.structure_decoder = BartForConditionalGeneration(bart_config)
    
        # Visual feature projection
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            config.feature_dim
        )
        
        # Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=config.feature_dim,
            dropout=config.dropout
        )
        
        # Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=config.temperature,
        )
        
        # Span projection
        self.span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        
        # Row와 Column을 위한 별도의 projection layers
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        images: torch.Tensor = batch['images']           # (B, 3, H, W)
        text_regions: torch.Tensor = batch['bboxes']     # (B, N, 4) - normalized bboxes
        labels: Optional[torch.Tensor] = batch['tokens']  # (B, L)
        attention_mask: Optional[torch.Tensor] = batch['attention_mask']  # (B, L)
        data_tag_mask: Optional[torch.Tensor] = batch['data_tag_mask']  # (B, L)
        
        B = images.size(0)
        N = text_regions.size(1)  # 실제 bbox 수
        
        # 1. Image Encoding (논문 3.2)
        visual_features = self.image_encoder(images).last_hidden_state  # (B, P, D)
        visual_features = self.visual_proj(visual_features)  # (B, P, D)
        
        # 2. Layout Encoding (논문 3.3)
        layout_embedding = self.layout_encoder(text_regions, visual_features)  # (B, N, D)
        
        # 3. Logical Structure Generation (논문 3.4)
        encoder_outputs = BaseModelOutput(
            last_hidden_state=visual_features,
            hidden_states=None,
            attentions=None
        )
        
        # Encoder attention mask
        encoder_attention_mask = torch.ones(
            B, visual_features.size(1), 
            dtype=torch.long, 
            device=visual_features.device
        )
        
        # Training & Validation: 동일한 방식으로 forward pass
        if labels is not None:
            # Position indices가 max_position_embeddings를 초과하지 않도록 보장
            if labels.size(1) > self.config.max_seq_length:
                labels = labels[:, :self.config.max_seq_length]
            
            # labels를 contiguous하게 만들어서 view 연산이 가능하도록 함
            labels = labels.contiguous()
            
            decoder_input_ids = shift_tokens_right(
                labels,
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id
            )
            
            attention_mask = attention_mask[:, :self.config.max_seq_length].contiguous()
            
            decoder_outputs = self.structure_decoder(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
            
            tag_logits = decoder_outputs.logits.contiguous()
            last_hidden_state = decoder_outputs.decoder_hidden_states[-1].contiguous()
            
        
        # else:
        #     # Final Inference: generate 사용
        #     decoder_outputs = self.structure_decoder.generate(
        #         encoder_outputs=encoder_outputs.last_hidden_state,
        #         attention_mask=encoder_attention_mask,
        #         max_length=self.config.max_seq_length,
        #         num_beams=4,
        #         early_stopping=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         bos_token_id=self.tokenizer.bos_token_id,
        #         use_cache=True,
        #         output_hidden_states=True,
        #         return_dict_in_generate=True,
        #         output_scores=True
        #     )
        #     return {
        #         'sequences': decoder_outputs.sequences,
        #         'scores': decoder_outputs.scores
        #     }
        
        # 4. Layout Pointing (논문 3.5)
        # last_hidden_state에는 text_regions 정보가 포함되어 있어야 함
        last_hidden_state = torch.cat([
            layout_embedding,  # box features (B, N, D)
            last_hidden_state  # tag features (B, L, D)
        ], dim=1)
        
        pointer_logits, empty_logits = self.layout_pointer(
            decoder_hidden_states=last_hidden_state,
            num_boxes=N,
            data_tag_mask=data_tag_mask,
        )
        
        # Span-aware contrastive learning을 위한 similarity matrix 계산
        row_sim_matrix, col_sim_matrix = self.get_sim_matrix(layout_embedding)
        
        # Span coefficient 계산 (OTSL 토큰으로부터)
        # box_indices는 이제 (B, N, M) 형태이므로 flatten할 필요 없음
        row_span_coef, col_span_coef = get_coef_matrix(
            batch['tokens'], 
            self.tokenizer, 
            batch['box_indices'],  # (B, N, M)
            N
        )
        
        # 디버깅 정보를 파일에 저장
        # with open('matrices_debug.txt', 'w') as f:
        #     f.write("=== Debugging Information ===\n\n")
            
        #     # 1. Shape 정보
        #     f.write("Shape Information:\n")
        #     f.write(f"layout_embedding: {layout_embedding.shape}\n")
        #     f.write(f"row_sim_matrix: {row_sim_matrix.shape}\n")
        #     f.write(f"col_sim_matrix: {col_sim_matrix.shape}\n")
        #     f.write(f"row_span_coef: {row_span_coef.shape}\n")
        #     f.write(f"col_span_coef: {col_span_coef.shape}\n\n")
            
        #     # 2. 값 범위 통계
        #     f.write("Value Statistics:\n")
        #     f.write("Row Similarity Matrix:\n")
        #     f.write(f"  - min: {row_sim_matrix.min().item():.4f}\n")
        #     f.write(f"  - max: {row_sim_matrix.max().item():.4f}\n")
        #     f.write(f"  - mean: {row_sim_matrix.mean().item():.4f}\n")
        #     f.write(f"  - std: {row_sim_matrix.std().item():.4f}\n")
        #     f.write(f"  - non-zero elements: {(row_sim_matrix != 0).sum().item()}\n\n")
            
        #     f.write("Column Similarity Matrix:\n")
        #     f.write(f"  - min: {col_sim_matrix.min().item():.4f}\n")
        #     f.write(f"  - max: {col_sim_matrix.max().item():.4f}\n")
        #     f.write(f"  - mean: {col_sim_matrix.mean().item():.4f}\n")
        #     f.write(f"  - std: {col_sim_matrix.std().item():.4f}\n")
        #     f.write(f"  - non-zero elements: {(col_sim_matrix != 0).sum().item()}\n\n")
            
        #     f.write("Row Span Coefficient:\n")
        #     f.write(f"  - min: {row_span_coef.min().item():.4f}\n")
        #     f.write(f"  - max: {row_span_coef.max().item():.4f}\n")
        #     f.write(f"  - mean: {row_span_coef.mean().item():.4f}\n")
        #     f.write(f"  - std: {row_span_coef.std().item():.4f}\n")
        #     f.write(f"  - non-zero elements: {(row_span_coef != 0).sum().item()}\n\n")
            
        #     f.write("Column Span Coefficient:\n")
        #     f.write(f"  - min: {col_span_coef.min().item():.4f}\n")
        #     f.write(f"  - max: {col_span_coef.max().item():.4f}\n")
        #     f.write(f"  - mean: {col_span_coef.mean().item():.4f}\n")
        #     f.write(f"  - std: {col_span_coef.std().item():.4f}\n")
        #     f.write(f"  - non-zero elements: {(col_span_coef != 0).sum().item()}\n\n")
            
        #     # 3. OTSL 토큰과 box indices 정보
        #     f.write("OTSL and Box Indices Info:\n")
        #     f.write(f"Number of boxes (N): {N}\n")
        #     f.write(f"Box indices shape: {batch['box_indices'].shape}\n")
        #     f.write(f"Box indices sample: {batch['box_indices'][0].tolist()[:10]}\n")
            
        #     # 첫 번째 샘플의 OTSL 토큰 출력
        #     tokens = [self.tokenizer.id2token[tid.item()] for tid in batch['tokens'][0] if tid.item() in self.tokenizer.id2token]
        #     f.write(f"OTSL tokens sample: {' '.join(tokens[:tokens.index('[EOS]')])}...\n\n")
            
        #     # 4. 실제 행렬 값 샘플 (첫 번째 배치의 5x5 부분)
        #     f.write("Matrix Values Sample (5x5):\n")
        #     f.write("Row Similarity Matrix:\n")
        #     f.write(f"{row_sim_matrix[0, :5, :5]}\n\n")
        #     f.write("Row Span Coefficient:\n")
        #     f.write(f"{row_span_coef[0, :5, :5]}\n\n")
            
        #     f.write("Column Similarity Matrix:\n")
        #     f.write(f"{col_sim_matrix[0, :5, :5]}\n\n")
        #     f.write("Column Span Coefficient:\n")
        #     f.write(f"{col_span_coef[0, :5, :5]}\n\n")
        
        # 5. Return outputs for loss calculation
        outputs = {
            'tag_logits': tag_logits,                # (B, L, V) - V는 vocab size
            'box_features': layout_embedding,         # (B, N, D) - N은 실제 bbox 수
            'tag_features': last_hidden_state[:, N:], # (B, L, D)
            'pointer_logits': pointer_logits,         # (B, N, L) - bbox와 token 간의 관계
            'empty_logits': empty_logits,            # (B, L) - empty cell 예측
            'data_tag_mask': data_tag_mask,          # (B, L) - 'C' 토큰의 위치
            'row_sim_matrix': row_sim_matrix,        # (B, N, N) - bbox 간의 row-wise similarity
            'col_sim_matrix': col_sim_matrix,        # (B, N, N) - bbox 간의 column-wise similarity
            'row_span_coef': row_span_coef,          # (B, N, N) - bbox 간의 row-wise span coefficient
            'col_span_coef': col_span_coef           # (B, N, N) - bbox 간의 column-wise span coefficient
        }
        
        return outputs

    def get_sim_matrix(self, layout_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity matrices for row-wise and column-wise contrastive learning"""
        # 1. Project layout embeddings with different projections (Equation 4)
        row_projected_features = self.row_span_proj(layout_embedding)  # (B, N, D)
        col_projected_features = self.col_span_proj(layout_embedding)  # (B, N, D)
        
        # 2. Compute similarity matrices (dot product)
        row_sim_matrix = torch.matmul(
            row_projected_features, 
            row_projected_features.transpose(-2, -1)
        ) # (B, N, N)
        
        col_sim_matrix = torch.matmul(
            col_projected_features, 
            col_projected_features.transpose(-2, -1)
        )  # (B, N, N)
        
        row_sim_matrix = F.normalize(row_sim_matrix, dim=-1)
        col_sim_matrix = F.normalize(col_sim_matrix, dim=-1)
        
        return row_sim_matrix, col_sim_matrix