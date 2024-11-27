from typing import Dict, Optional, Union, Any
import torch
import torch.nn as nn
from transformers import SwinModel, BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
import torch.nn.functional as F

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
        
        if not inference_mode:
            # 학습 모드: pretrained 모델 로드
            self.image_encoder = SwinModel.from_pretrained(config.swin_model_name)
            
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
        else:
            # 추론 모드: 빈 모델 생성 (가중치는 나중에 로드됨)
            self.image_encoder = SwinModel(self.image_encoder.config)
            self.structure_decoder = BartForConditionalGeneration(self.structure_decoder.config)
        
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
            tokenizer=self.tokenizer
        )
        
        # Span projection
        self.span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        
        # Span-aware contrastive projection layer 추가
        self.span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.temperature = config.temperature

    def forward(
        self,
        images: torch.Tensor,           # (B, 3, H, W)
        text_regions: torch.Tensor,     # (B, N, 4) - normalized bboxes
        labels: Optional[torch.Tensor] = None,  # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        row_spans: Optional[torch.Tensor] = None,  # (B, N, N)
        col_spans: Optional[torch.Tensor] = None   # (B, N, N)
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        B = images.size(0)
        N = text_regions.size(1)
        
        # 1. Image Encoding (논문 3.2)
        visual_features = self.image_encoder(images).last_hidden_state  # {zi}
        visual_features = self.visual_proj(visual_features)
        
        # 2. Layout Encoding (논문 3.3)
        layout_embedding = self.layout_encoder(text_regions, visual_features)  # {lj}
        
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
            
            # attention mask도 같이 자르고 contiguous하게 만듦
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.max_seq_length].contiguous()
            else:
                attention_mask = torch.ones_like(labels, dtype=torch.long)
            
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
        
        else:
            # Final Inference: generate 사용
            decoder_outputs = self.structure_decoder.generate(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=encoder_attention_mask,
                max_length=self.config.max_seq_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                use_cache=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            return {
                'sequences': decoder_outputs.sequences,
                'scores': decoder_outputs.scores
            }
        
        # 4. Layout Pointing (논문 3.5)
        # last_hidden_state에는 text_regions 정보가 포함되어 있어야 함
        last_hidden_state = torch.cat([
            layout_embedding,  # box features (B, N, D)
            last_hidden_state  # tag features (B, L, D)
        ], dim=1)
        
        pointer_logits, empty_logits, data_tag_mask = self.layout_pointer(
            decoder_hidden_states=last_hidden_state,
            num_boxes=N
        )
        
        # 5. Return outputs for loss calculation
        outputs = {
            'tag_logits': tag_logits,                # (B, L, V)
            'box_features': layout_embedding,        # (B, N, D)
            'tag_features': last_hidden_state[:, N:],  # (B, L, D)
            'pointer_logits': pointer_logits,        # (B, N, T)
            'empty_logits': empty_logits,           # (B, T)
            'data_tag_mask': data_tag_mask          # (B, T)
        }
        
        # Span-aware contrastive features 계산
        if row_spans is not None and col_spans is not None:
            box_features = layout_embedding  # (B, N, D)
            projected_features = self.span_proj(box_features)  # (B, N, D)
            
            # Row-wise contrastive features
            row_overlap = torch.matmul(row_spans, row_spans.transpose(-2, -1))  # (B, N, N)
            row_span_coef = row_overlap / (
                torch.sum(row_spans, dim=-1, keepdim=True) * 
                torch.sum(row_spans, dim=-1, keepdim=True).transpose(-2, -1)
            )  # Equation (6)
            
            # Column-wise contrastive features
            col_overlap = torch.matmul(col_spans, col_spans.transpose(-2, -1))  # (B, N, N)
            col_span_coef = col_overlap / (
                torch.sum(col_spans, dim=-1, keepdim=True) * 
                torch.sum(col_spans, dim=-1, keepdim=True).transpose(-2, -1)
            )  # Equation (6)
            
            outputs.update({
                'projected_features': projected_features,  # (B, N, D)
                'row_span_coef': row_span_coef,          # (B, N, N)
                'col_span_coef': col_span_coef           # (B, N, N)
            })
        
        return outputs