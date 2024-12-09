from typing import Dict, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
from transformers import SwinModel, Swinv2Model, BartModel, BartConfig
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
        self.image_encoder = SwinModel.from_pretrained(config.swin_model_name)
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            config.feature_dim
        )
        
        # 2. Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=config.feature_dim,
            dropout=config.dropout,
            input_size=config.image_size
        )
        self.layout_pos_embed = nn.Embedding(config.total_sequence_length, config.feature_dim)
        self.prompt_layer_norm = nn.LayerNorm(config.feature_dim)
        
        # 3. BART Decoder 초기화
        self.init_bart_decoder(config)
        
        # 4. Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=config.temperature,
        )
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

    def init_bart_decoder(self, config):
        """Pre-trained BART 초기화"""
        # Pre-trained BART 로드
        pretrained_bart = BartModel.from_pretrained('facebook/bart-base')
        
        # Config 수정
        bart_config = pretrained_bart.config
        bart_config.vocab_size = self.tokenizer.vocab_size
        bart_config.max_position_embeddings = config.total_sequence_length
        bart_config.d_model = config.feature_dim
        bart_config.is_encoder_decoder = False
        bart_config.add_cross_attention = True
        
        # BART 모델 초기화 및 pre-trained 가중치 복사
        self.bart = BartModel(bart_config)
        
        # Decoder 가중치 복사 (크기가 맞는 부분만)
        pretrained_state = pretrained_bart.decoder.state_dict()
        decoder_state = self.bart.decoder.state_dict()
        
        for name, param in pretrained_state.items():
            if name in decoder_state and decoder_state[name].shape == param.shape:
                decoder_state[name].copy_(param)
        
        # Output projection 초기화
        self.output_projection = nn.Linear(config.feature_dim, self.tokenizer.vocab_size, bias=False)
        
        # OTSL 토큰 임베딩 특별 초기화
        with torch.no_grad():
            std = self.bart.shared.weight.std()
            for token_id in self.tokenizer.otsl_token_ids:
                self.bart.shared.weight[token_id].normal_(0, std)

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
        labels = batch['token_ids']        
        attention_mask = batch['attention_mask']
        B = images.size(0)
        
        # 1. Image Encoding
        encoder_outputs = self.image_encoder(
            images,
            output_hidden_states=True
        )
        visual_features = encoder_outputs.last_hidden_state
        # Reshape & normalize visual features
        H = W = int(visual_features.size(1) ** 0.5)
        visual_features = visual_features.view(B, H, W, -1).permute(0, 3, 1, 2)
        visual_features = F.normalize(visual_features, p=2, dim=1)
        visual_features = self.visual_proj(visual_features.permute(0, 2, 3, 1))
        visual_features = visual_features.reshape(B, H*W, -1)

        # 2. Layout encoding & prompt preparation
        layout_embedding = self.layout_encoder(text_regions, visual_features)
        layout_prompt = self.prepare_layout_prompt(layout_embedding)

        visual_attention_mask = attention_mask[:, :visual_features.size(1)]

        # 4. Prepare decoder inputs
        decoder_input_ids = shift_tokens_right(
            labels,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id
        )

        # 1. Prepare inputs following Donut's approach
        token_embeds = self.bart.decoder.embed_tokens(decoder_input_ids)  # token embeddings
        prompt_inputs = torch.cat([
                        layout_prompt,     # 먼저 layout context를 prompt로 제공
                        token_embeds      # 그 다음 실제 디코딩할 토큰
                    ], dim=1)

        # 1. Attention masks (from Donut)
        prompt_length = layout_prompt.size(1)
        total_length = prompt_length + self.otsl_sequence_length
        
        # Donut style: 기본 attention mask는 2D
        attention_mask = torch.zeros(
            (B, total_length),
            device=decoder_input_ids.device
        )
        
        # TFLOP style: layout 정보를 활용한 causal masking
        if not self.inference_mode:
            causal_mask = torch.triu(
                torch.full((self.otsl_sequence_length, self.otsl_sequence_length), -1e4),
                diagonal=1
            ).to(decoder_input_ids.device)
            
            # Token sequence 부분만 causal하게 제한
            attention_mask[:, prompt_length:] = causal_mask[0]
        
        # 2. Decoder forward pass (from Donut)
        decoder_outputs = self.bart.decoder(
            inputs_embeds=prompt_inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=visual_attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        # 3. Get decoder outputs
        last_hidden_state = decoder_outputs.last_hidden_state
        bbox_embeddings = last_hidden_state[:, :layout_prompt.size(1), :]      # b_j            # (B, 688, 1024)
        logical_structure_embeddings = last_hidden_state[:, layout_prompt.size(1):, :]  # t_k   # (B, 688, 1024)

        tag_logits = self.output_projection(logical_structure_embeddings)


        pointer_logits, empty_pointer_logits = self.layout_pointer(
            box_features=bbox_embeddings,
            tag_features=logical_structure_embeddings
        )

        row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
            layout_prompt,
            attention_mask=attention_mask[:, :self.layout_prompt_length]
        )
        
        row_span_coef, col_span_coef, shapes = get_coef_matrix(
            labels,
            self.tokenizer, 
            self.layout_prompt_length
        )

        outputs = {
            'tag_logits': tag_logits,
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_pointer_logits,
            'row_sim_matrix': row_sim_matrix,
            'col_sim_matrix': col_sim_matrix,
            'row_span_coef': row_span_coef ,
            'col_span_coef': col_span_coef
        }
        
        return outputs

        # TODO: inference 구현
        
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
