from typing import Dict, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import SwinModel, BartModel, BartConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
import torch.nn.functional as F
from utils.util import get_coef_matrix
import math
from .bart_decoder import TFLOPDecoder

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
            otsl_sequence_length=config.total_sequence_length // 2
        )
        
        # tokenzier에 종속되게 설정        
        self.layout_prompt_length = self.otsl_sequence_length = self.tokenizer.otsl_sequence_length
        
        # 1. Image Encoder (Swin Transformer)
        self.image_encoder = SwinModel.from_pretrained(config.swin_model_name)
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            config.feature_dim
        )
        
        # 2. Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=config.feature_dim,
            dropout=config.dropout
        )
        self.layout_pos_embed = nn.Embedding(config.total_sequence_length, config.feature_dim)
        self.prompt_layer_norm = nn.LayerNorm(config.feature_dim)
        self.layout_prompt_norm = nn.LayerNorm(config.feature_dim)
        
        # 3. Logical Structure Decoder (BART decoder-only)
        bart_config = BartConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=config.total_sequence_length,
            d_model=config.feature_dim,
            decoder_layers=config.decoder_layers,
            decoder_attention_heads=config.decoder_attention_heads,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            is_encoder_decoder=False,  # decoder-only 구조 사용
            add_cross_attention=True,  # cross-attention 활성화
        )
        self.bart = BartModel(bart_config)
        self.bart.decoder = TFLOPDecoder(bart_config)
        self.output_projection = nn.Linear(config.feature_dim, self.tokenizer.vocab_size, bias=False)
        
        # 4. Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=config.temperature,
        )
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

    def prepare_layout_prompt(
        self,
        layout_embedding: torch.Tensor,  # (B, :688, 1024)
    ) -> torch.Tensor:
        """Layout prompt 준비
        
        Positional Encoding 방식 적용해서 layout prompt 생성
        
        Args:
            layout_embedding: Layout encoder의 출력
            
        Returns:
            layout_prompt: Context prompt로 사용될 layout embeddings
        """
        B, N, D = layout_embedding.shape       # (B, 688, 1024)

        # Add position embeddings to layout embeddings (실제 box 수에 맞춰서)
        position_ids = torch.arange(N, device=layout_embedding.device)  # (688)
        position_embeddings = self.layout_pos_embed(position_ids)  # (688, 1024)
        
        # Combine layout and position information
        layout_prompt = layout_embedding + position_embeddings.unsqueeze(0)  # (B, 688, 1024)
        
        # Layer normalization
        layout_prompt = self.prompt_layer_norm(layout_prompt)
        
        return layout_prompt

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        images = batch['images']           # (B, 3, 768, 738)
        text_regions = batch['bboxes']     # (B, :688, 4)
        labels = batch['token_ids']         # (B, :688)
        attention_mask = batch['attention_mask']  # (B, 1376)
        data_tag_mask = batch['data_tag_mask']   # (B, 1376)
        
        B = images.size(0)
        
        # 1. Image Encoding
        visual_features = self.image_encoder(images).last_hidden_state
        visual_features = self.visual_proj(visual_features)
        
        # 2. Layout Encoding - 실제 box 수만큼만 사용
        layout_embedding = self.layout_encoder(
            text_regions,       # (B, :688, 4)
            visual_features     # (B, 576, 1024) -> Swin T 출력
        )
        # Layout prompt 준비 - 실제 box 수 기준
        layout_prompt = self.prepare_layout_prompt(layout_embedding)    # (B, 688, 1024)

        if labels is not None:
            # Token embeddings
            token_embeddings = self.bart.shared(labels)  # (B, 688, 1024)
            
            # Attention masks 생성
            # 1. Causal mask 생성
            causal_mask = self._make_causal_mask(
                seq_length=labels.size(1),                          # otsl_sequence_length = 688
                layout_prompt_length=self.layout_prompt_length,
                device=labels.device,
                batch_size=B
            )  # (B, 1, 1376, 1376)
            
            # 2. Padding mask 적용
            padding_mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 1376)
            token_attention_mask = causal_mask * padding_mask  # broadcasting
        
            # Visual attention mask
            visual_attention_mask = torch.ones((B, visual_features.size(1)), device=visual_features.device)  # (B, 576)
            
            decoder_inputs = torch.cat([token_embeddings, layout_prompt], dim=1)  # (B, 1376, 1024)
            
            # Decoder forward pass
            decoder_outputs = self.bart.decoder(
                inputs_embeds=decoder_inputs,
                attention_mask=token_attention_mask,  # 패딩 마스크가 적용된 causal mask
                layout_prompt=layout_prompt,
                visual_features=visual_features,
                encoder_attention_mask=visual_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden_state = decoder_outputs.last_hidden_state   # (B, 1376, 1024)
            box_features = last_hidden_state[:, :self.layout_prompt_length, :]  # (B, 688, 1024)
            tag_features = last_hidden_state[:, self.layout_prompt_length:, :]  # (B, 688, 1024)
            tag_logits = self.output_projection(tag_features)        # (B, 688, Vocab_size)
            
            
            box_proj, tag_proj, empty_proj = self.layout_pointer(
                decoder_hidden_states=last_hidden_state,
                layout_prompt_length=self.layout_prompt_length
            ) # (B, :688, 1024), (B, 688:, 1024), (B, 1, 1024)

            # Contrastive learning을 위한 similarity matrix 계산 (실제 box 수 기준)
            row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
                box_features,
                attention_mask = attention_mask[:, :self.layout_prompt_length]
            )
            
            # # Span coefficient 계산 (실제 box 수 기준)
            row_span_coef, col_span_coef = get_coef_matrix(
                labels, # (B, 688)
                self.tokenizer, 
                self.layout_prompt_length            # (B, 688, M)
            )
                        
            # Return outputs for loss calculation
            outputs = {
                'tag_logits': tag_logits,                # (B, 688, 9) - vocabulary logits
                'box_proj': box_proj,                    # (B, :688, 1024) - decoder의 box features
                'tag_proj': tag_proj,                    # (B, 688:, 1024) - decoder의 tag features
                'empty_proj': empty_proj,                # (B, 1, 1024) - empty cell embedding
                'data_tag_mask': data_tag_mask,          # (B, 1376)
                'row_sim_matrix': row_sim_matrix,        # (B, 688, 688) - row-wise similarity
                'col_sim_matrix': col_sim_matrix,        # (B, 688, 688) - column-wise similarity
                'row_span_coef': row_span_coef,          # (B, 688, 688) - row span coefficients
                'col_span_coef': col_span_coef           # (B, 688, 688) - column span coefficients
            }
            
            return outputs

        # TODO: inference 구현
        
    def get_sim_matrix(self, box_features: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity matrices for row-wise and column-wise contrastive learning
        
        Args:
            box_features: logical structure decoder의 출력 (B, N, D)
            
        Returns:
            row_sim_matrix: row-wise similarity matrix (B, N, N)
            col_sim_matrix: column-wise similarity matrix (B, N, N)
        """
        
        # 1. Project layout embeddings (Equation 4: b̂j = projs(bj))
        row_projected = self.row_span_proj(box_features)  # (B, N, D)
        col_projected = self.col_span_proj(box_features)  # (B, N, D)
        
        # 2. L2 normalize projected features
        row_projected = F.normalize(row_projected, p=2, dim=-1)  # (B, N, D)
        col_projected = F.normalize(col_projected, p=2, dim=-1)  # (B, N, D)
        
        # 3. Compute similarity matrices
        row_sim_matrix = torch.matmul(
            row_projected, 
            row_projected.transpose(-2, -1)
        )  # (B, N, N)
        
        col_sim_matrix = torch.matmul(
            col_projected, 
            col_projected.transpose(-2, -1)
        )  # (B, N, N)
        
        # 패딩 마스크 적용 
        mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, N, N)
        row_sim_matrix = row_sim_matrix * mask
        col_sim_matrix = col_sim_matrix * mask
        
        return row_sim_matrix, col_sim_matrix

    def _make_causal_mask(
        self,
        seq_length: int,          # otsl_sequence_length (688)
        layout_prompt_length: int, # layout_prompt_length (688)
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:
        """
        Layout prompt와 OTSL sequence를 위한 causal mask 생성
        전체 sequence length = layout_prompt_length + otsl_sequence_length
        """
        total_length = layout_prompt_length + seq_length  # 1376
        
        # 전체 크기의 mask 생성
        mask = torch.full((total_length, total_length), -1e4, device=device)
        mask = torch.triu(mask, diagonal=1)
        
        # Layout prompt 부분은 모든 위치에서 참조 가능
        mask[:, :layout_prompt_length] = 0
        
        # OTSL sequence 부분의 causal attention
        otsl_mask = torch.triu(torch.ones(seq_length, seq_length, device=device))
        mask[layout_prompt_length:, layout_prompt_length:] = otsl_mask * -1e4
        
        return mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_length, total_length)  # (B, 1, 1376, 1376)