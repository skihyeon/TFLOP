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
            total_sequence_length=config.total_sequence_length
        )
        
        # Image Encoder (Swin Transformer)
        self.image_encoder = SwinModel.from_pretrained(config.swin_model_name)
        
        # Visual feature projection
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            config.feature_dim
        )
        
        # BART decoder-only 설정 및 초기화
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
        
        # BartModel 초기화 (decoder만 사용)
        self.bart = BartModel(bart_config)
        self.output_projection = nn.Linear(config.feature_dim, self.tokenizer.vocab_size, bias=False)
        
        # Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=config.feature_dim,
            dropout=config.dropout
        )
        
        # Layout prompt를 위한 추가 components
        self.layout_pos_embed = nn.Embedding(config.total_sequence_length, config.feature_dim)
        self.prompt_layer_norm = nn.LayerNorm(config.feature_dim)
        
        # Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=config.temperature,
        )
        
        # Span projections
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

        self.layout_prompt_norm = nn.LayerNorm(config.feature_dim)
        
        # Token embedding scale factor 추가 
        self.embed_scale = math.sqrt(config.feature_dim) if config.scale_embedding else 1.0
        
        # BART decoder를 custom decoder로 교체
        self.bart.decoder = TFLOPDecoder(bart_config)

    def prepare_layout_prompt(
        self,
        layout_embedding: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        """Layout prompt 준비
        
        Args:
            layout_embedding: Layout encoder의 출력
            
        Returns:
            layout_prompt: Context prompt로 사용될 layout embeddings
        """
        B, N, D = layout_embedding.shape
        
        # Add position embeddings to layout embeddings (실제 box 수에 맞춰서)
        position_ids = torch.arange(N, device=layout_embedding.device)
        position_embeddings = self.layout_pos_embed(position_ids)
        
        # Combine layout and position information
        layout_prompt = layout_embedding + position_embeddings.unsqueeze(0)
        
        # Layer normalization
        layout_prompt = self.prompt_layer_norm(layout_prompt)
        
        return layout_prompt

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        
        images = batch['images']           # (B, 3, H, W)
        text_regions = batch['bboxes']     # (B, max_boxes, 4)
        labels = batch['tokens']           # (B, L)
        attention_mask = batch['attention_mask']  # (B, total_seq_len)
        data_tag_mask = batch['data_tag_mask']   # (B, total_seq_len)
        num_boxes = batch['num_boxes']     # (B,) - 실제 box 수
        
        B = images.size(0)
        N = text_regions.size(1)  # max_boxes
        
        # print(f"\nTFLOP forward:")
        # print(f"  B: {B}, N: {N}")
        # print(f"  text_regions shape: {text_regions.shape}")
        # print(f"  labels shape: {labels.shape}")
        # print(f"  num_boxes: {num_boxes}")
        # print(f"  attention_mask shape: {attention_mask.shape}")
        # print(f"  data_tag_mask shape: {data_tag_mask.shape}")
        
        # 1. Image Encoding
        visual_features = self.image_encoder(images).last_hidden_state
        visual_features = self.visual_proj(visual_features)
        
        # 2. Layout Encoding - 실제 box 수만큼만 사용
        layout_embedding = self.layout_encoder(
            text_regions[:, :num_boxes.max()],  # 배치에서 가장 큰 box 수까지만
            visual_features
        )
        
        # Layout prompt 준비 - 실제 box 수 기준
        layout_prompt = self.prepare_layout_prompt(layout_embedding)
        
        if labels is not None:
            # Token embeddings
            token_embeddings = self.bart.shared(labels) * self.embed_scale
            
            # Attention masks - 실제 길이 기준
            token_attention_mask = self._make_causal_mask(
                seq_length=labels.size(1),
                layout_prompt_length=num_boxes.max(),  # 배치의 최대 box 수
                device=labels.device,
                batch_size=B
            )
            
            # Visual attention mask
            visual_attention_mask = torch.ones(
                (B, visual_features.size(1)),
                device=visual_features.device
            )
            
            # Decoder forward pass
            decoder_outputs = self.bart.decoder(
                inputs_embeds=token_embeddings,
                attention_mask=token_attention_mask,
                layout_prompt=layout_prompt,
                visual_features=visual_features,
                encoder_attention_mask=visual_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden_state = decoder_outputs.last_hidden_state
            # print(f"Decoder last hidden state shape: {last_hidden_state.shape}, range: [{last_hidden_state.min():.4f}, {last_hidden_state.max():.4f}]")
            # Decoder hidden states range : [-5, 5] -> 적절한 값
            
            # 전체 sequence에 대해 vocabulary projection 수행
            logits = self.output_projection(last_hidden_state)  # (B, N+L, V)
            # print(f"  logits shape: {logits.shape}")
            
            # Layout prompt 부분을 제외하고 고정된 위치에서 자르기
            max_boxes = num_boxes.max()
            tag_logits = logits[:, max_boxes:, :]  # (B, L, V)
            # print(f"  tag_logits shape: {tag_logits.shape}")
            
            # Layout pointer 처리 (실제 box 수 기준)
            # pointer_logits, empty_logits = self.layout_pointer(
            #     decoder_hidden_states=last_hidden_state,
            #     num_boxes=num_boxes.max()
            # )
            
            box_proj, tag_proj, empty_proj = self.layout_pointer(
                decoder_hidden_states=last_hidden_state,
                num_boxes=num_boxes.max()
            )
            # print(f"Pointer logits shape: {pointer_logits.shape}, range: [{pointer_logits.min():.4f}, {pointer_logits.max():.4f}]")
            # print(f"Empty logits shape: {empty_logits.shape}, range: [{empty_logits.min():.4f}, {empty_logits.max():.4f}]")
            
            # Contrastive learning을 위한 similarity matrix 계산 (실제 box 수 기준)
            row_sim_matrix, col_sim_matrix = self.get_sim_matrix(layout_embedding)
            
            # Span coefficient 계산 (실제 box 수 기준)
            row_span_coef, col_span_coef = get_coef_matrix(
                batch['tokens'], 
                self.tokenizer, 
                batch['box_indices'][:, :num_boxes.max()],  # 실제 box 수까지만
                num_boxes.max()
            )
            
            # Return outputs for loss calculation
            outputs = {
                'tag_logits': tag_logits,                # (B, L, V) - vocabulary logits
                'box_proj': box_proj,                    # (B, N, D) - decoder의 box features
                'tag_proj': tag_proj,                    # (B, T, D) - decoder의 tag features
                'empty_proj': empty_proj,                # (B, D) - empty cell embedding
                'data_tag_mask': data_tag_mask,          # (B, total_seq_len)
                'row_sim_matrix': row_sim_matrix,        # (B, N, N) - row-wise similarity
                'col_sim_matrix': col_sim_matrix,        # (B, N, N) - column-wise similarity
                'row_span_coef': row_span_coef,          # (B, N, N) - row span coefficients
                'col_span_coef': col_span_coef           # (B, N, N) - column span coefficients
            }
            
            return outputs
        
        else:  # Inference mode
            # Auto-regressive generation을 위한 초기 입력
            decoder_input_ids = torch.full(
                (B, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=images.device
            )
            
            # Generation configs
            max_length = self.config.total_sequence_length - num_boxes.max()
            generated_tokens = []
            past_key_values = None
            
            # Auto-regressive generation
            for _ in range(max_length):
                # Token embeddings
                token_embeddings = self.bart.shared(decoder_input_ids) * self.embed_scale
                
                # Decoder forward pass
                decoder_outputs = self.bart.decoder(
                    inputs_embeds=token_embeddings,
                    layout_prompt=layout_prompt,
                    visual_features=visual_features,
                    encoder_attention_mask=visual_attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True
                )
                
                # Get logits and past key values
                last_hidden = decoder_outputs.last_hidden_state
                logits = self.output_projection(last_hidden)
                past_key_values = decoder_outputs.past_key_values
                
                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1)
                generated_tokens.append(next_token)
                
                # Break if EOS token is generated
                if next_token == self.tokenizer.eos_token_id:
                    break
                    
                # Update decoder input ids
                decoder_input_ids = next_token.unsqueeze(-1)
            
            # Process generated sequence
            generated_sequence = torch.stack(generated_tokens, dim=1)
            
            # Get layout pointer outputs for the full sequence
            last_hidden_state = decoder_outputs.last_hidden_state
            pointer_logits, empty_logits = self.layout_pointer(
                decoder_hidden_states=last_hidden_state,
                num_boxes=num_boxes.max()
            )
            
            return {
                'generated_sequence': generated_sequence,  # (B, L)
                'pointer_logits': pointer_logits,         # (B, N, L)
                'empty_logits': empty_logits             # (B, L)
            }

    def get_sim_matrix(self, layout_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute similarity matrices for row-wise and column-wise contrastive learning"""
        # Project layout embeddings
        row_projected_features = self.row_span_proj(layout_embedding)  # (B, N, D)
        col_projected_features = self.col_span_proj(layout_embedding)  # (B, N, D)
        
        # Compute similarity matrices
        row_sim_matrix = torch.matmul(
            row_projected_features, 
            row_projected_features.transpose(-2, -1)
        )  # (B, N, N)
        
        col_sim_matrix = torch.matmul(
            col_projected_features, 
            col_projected_features.transpose(-2, -1)
        )  # (B, N, N)
        
        # Normalize
        row_sim_matrix = F.normalize(row_sim_matrix, dim=-1)
        col_sim_matrix = F.normalize(col_sim_matrix, dim=-1)
        
        return row_sim_matrix, col_sim_matrix

    def _make_causal_mask(
        self,
        seq_length: int,
        layout_prompt_length: int,
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:
        """
        Layout prompt를 고려한 causal mask 생성
        - Layout prompt는 모든 위치에서 참조 가능
        - Content 토큰들은 이전 토큰들과 layout prompt만 참조 가능
        
        Returns:
            mask: (batch_size, 1, seq_length, seq_length) 형태의 attention mask
                 0: 참조 가능, -1e4: 참조 불가능
        """
        # 기본 causal mask 생성
        mask = torch.full((seq_length, seq_length), -1e4, device=device)
        mask = torch.triu(mask, diagonal=1)
        
        # Layout prompt 부분은 모든 위치에서 참조 가능하도록 수정
        mask[:, :layout_prompt_length] = 0
        
        # Content 토큰들 간의 attention 강화
        content_mask = torch.triu(torch.ones(seq_length - layout_prompt_length, 
                                           seq_length - layout_prompt_length, 
                                           device=device))
        mask[layout_prompt_length:, layout_prompt_length:] = content_mask * -1e4
        
        # (batch_size, 1, seq_length, seq_length) 형태로 확장
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_length, seq_length)
        
        return mask