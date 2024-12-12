from typing import Dict, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import SwinModel, Swinv2Model, BartModel, BartConfig, AutoImageProcessor, SwinConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
import torch.nn.functional as F
from utils.util import get_coef_matrix
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
import logging

class TFLOP(nn.Module):
    def __init__(self, config: Any, inference_mode: bool = False) -> None:
        super().__init__()
        self.config = config
        self.inference_mode = inference_mode
        # Tokenizer 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=config.total_sequence_length // 2
        )
        
        # tokenizer에 종속되게 설정        
        self.layout_prompt_length = self.otsl_sequence_length = self.tokenizer.otsl_sequence_length
        
        # 1. Image Encoder (Swin Transformer)
        self.swin_config = SwinConfig.from_pretrained(config.swin_model_name)
        self.swin_config.image_size = config.image_size
        self.image_encoder = SwinModel.from_pretrained(config.swin_model_name, config=self.swin_config)
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            config.feature_dim
        )
        
        # 2. Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=config.feature_dim,
            dropout=getattr(config, 'dropout', 0.1),
            input_size=config.image_size
        )
        self.layout_pos_embed = nn.Embedding(config.total_sequence_length, config.feature_dim)
        self.prompt_layer_norm = nn.LayerNorm(config.feature_dim)
        
        # 3. BART Decoder 초기화
        self.bart_config = BartConfig.from_pretrained("facebook/bart-base")
        self.bart_config.vocab_size = self.tokenizer.vocab_size
        self.bart_config.d_model = config.feature_dim
        self.bart_config.add_cross_attention = True

        self.bart = BartModel(self.bart_config)
        self.output_projection = nn.Linear(config.feature_dim, self.tokenizer.vocab_size, bias=False)

        # 4. Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=getattr(config, 'temperature', 0.1),
        )
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

    def prepare_layout_prompt(
        self,
        layout_embedding: torch.Tensor,  # (B, :688, 1024)
    ) -> torch.Tensor:
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
        images = batch['images']           
        text_regions = batch['bboxes']     
        
        B = len(images)
        
        # 1. Image Encoding
        encoder_outputs = self.image_encoder(
            images,
            output_hidden_states=True
        )
        visual_features = encoder_outputs.last_hidden_state
        visual_features = F.normalize(visual_features, p=2, dim=-1)
        visual_features = self.visual_proj(visual_features)

        # 2. Layout encoding & prompt preparation
        layout_embedding = self.layout_encoder(text_regions, visual_features)
        layout_prompt = self.prepare_layout_prompt(layout_embedding)

        if self.training:
            # Training mode: teacher forcing 사용
            labels = batch['token_ids']        
            
            decoder_input_ids = shift_tokens_right(
                labels,
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id
            )

            token_embeds = self.bart.decoder.embed_tokens(decoder_input_ids)
            prompt_inputs = torch.cat([
                layout_prompt,
                token_embeds
            ], dim=1)

        elif not self.inference_mode:
            # Validation mode: auto-regressive 생성하되 training과 같은 shape 유지
            labels = batch['token_ids']
            seq_length = labels.size(1)
            
            curr_ids = torch.full(
                (B, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=images.device
            )
            
            # 전체 시퀀스 길이만큼의 텐서 미리 할당
            all_token_embeds = torch.zeros(
                B,
                seq_length,
                self.bart_config.d_model,
                device=images.device
            )
            
            # Auto-regressive하게 토큰 생성
            for step in range(seq_length - 1):  # BOS 토큰 제외
                token_embeds = self.bart.decoder.embed_tokens(curr_ids)
                prompt_inputs = torch.cat([
                    layout_prompt,
                    token_embeds
                ], dim=1)
                
                decoder_outputs = self.bart.decoder(
                    inputs_embeds=prompt_inputs,
                    encoder_hidden_states=visual_features,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                last_hidden_state = decoder_outputs.last_hidden_state
                current_token_embedding = last_hidden_state[:, -1:]
                
                # 현재 스텝의 임베딩 저장
                if step < seq_length - 1:
                    all_token_embeds[:, step:step+1] = current_token_embedding
                
                next_token = self.output_projection(current_token_embedding).argmax(dim=-1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # 마지막 토큰에 대한 임베딩
            token_embeds = self.bart.decoder.embed_tokens(curr_ids)
            all_token_embeds[:, -1:] = self.bart.decoder.embed_tokens(curr_ids[:, -1:])
            
            # Training과 동일한 형태로 입력 구성
            prompt_inputs = torch.cat([
                layout_prompt,
                all_token_embeds
            ], dim=1)

        else:
            # Inference mode: 순수 생성만 수행
            curr_ids = torch.full(
                (B, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=images.device
            )
            
            max_length = self.tokenizer.otsl_sequence_length
            
            for step in range(max_length):
                token_embeds = self.bart.decoder.embed_tokens(curr_ids)
                prompt_inputs = torch.cat([
                    layout_prompt,
                    token_embeds
                ], dim=1)
                
                decoder_outputs = self.bart.decoder(
                    inputs_embeds=prompt_inputs,
                    encoder_hidden_states=visual_features,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                last_hidden_state = decoder_outputs.last_hidden_state
                current_token_embedding = last_hidden_state[:, -1:]
                
                tag_logits = self.output_projection(current_token_embedding)
                next_token = tag_logits.argmax(dim=-1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Inference 결과만 반환
            decoder_outputs = self.bart.decoder(
                inputs_embeds=prompt_inputs,
                encoder_hidden_states=visual_features,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )
            
            last_hidden_state = decoder_outputs.last_hidden_state
            bbox_embeddings = last_hidden_state[:, :layout_prompt.size(1), :]
            logical_structure_embeddings = last_hidden_state[:, layout_prompt.size(1):, :]
            
            tag_logits = self.output_projection(logical_structure_embeddings)
            pointer_logits, empty_pointer_logits = self.layout_pointer(
                box_features=bbox_embeddings,
                tag_features=logical_structure_embeddings
            )
            
            return {
                'tag_logits': tag_logits,
                'pointer_logits': pointer_logits,
                'empty_pointer_logits': empty_pointer_logits,
                'generated_ids': curr_ids
            }

        # Training/Validation 공통 부분
        decoder_outputs = self.bart.decoder(
            inputs_embeds=prompt_inputs,
            encoder_hidden_states=visual_features,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = decoder_outputs.last_hidden_state
        bbox_embeddings = last_hidden_state[:, :layout_prompt.size(1), :]
        logical_structure_embeddings = last_hidden_state[:, layout_prompt.size(1):, :]

        tag_logits = self.output_projection(logical_structure_embeddings)
        pointer_logits, empty_pointer_logits = self.layout_pointer(
            box_features=bbox_embeddings,
            tag_features=logical_structure_embeddings
        )

        ## debug간 사용 안함함
        # row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
        #     layout_prompt,
        #     attention_mask=attention_mask[:, :self.layout_prompt_length]
        # )
        
        # row_span_coef, col_span_coef, shapes = get_coef_matrix(
        #     labels,
        #     self.tokenizer, 
        #     self.layout_prompt_length
        # )

        return {
            'tag_logits': tag_logits,
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_pointer_logits,
            'row_sim_matrix': None,
            'col_sim_matrix': None,
            'row_span_coef': None,
            'col_span_coef': None
        }

    def get_sim_matrix(self, box_features: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
