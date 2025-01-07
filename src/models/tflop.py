from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import Swinv2Model, Swinv2Config, BartForCausalLM, BartConfig, BartModel
from transformers import SwinModel, SwinConfig
from transformers.models.bart.modeling_bart import shift_tokens_right
from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
from utils.util import get_coef_matrix

class TFLOP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = OTSLTokenizer(otsl_sequence_length=config.otsl_max_length)
        self.layout_prompt_length = config.total_sequence_length - config.otsl_max_length
        
        self.swin_setup()
        self.other_setup()
        self.bart_setup()
    
    def swin_setup(self):
        self.swin_config = Swinv2Config.from_pretrained(self.config.swin_model_name)
        self.swin_config.image_size = self.config.image_size
        self.image_encoder = Swinv2Model.from_pretrained(self.config.swin_model_name, config=self.swin_config)
        self.visual_proj = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size, self.config.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.feature_dim * 2, self.config.feature_dim),
            nn.LayerNorm(self.config.feature_dim)
        )
        
    def bart_setup(self):
        self.bart_config = BartConfig(
            is_decoder=True,
            is_encoder_decoder=False,
            add_cross_attention=True,
            decoder_layers=4, 
            
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.feature_dim,
            max_position_embeddings=self.config.total_sequence_length,
            hidden_size=self.config.feature_dim,
            
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            activation_function='gelu',
            dropout=0.1,
            attention_dropout=0.1,
        )
        
        self.bart = BartForCausalLM(self.bart_config)
        
        
    def other_setup(self):
        # Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=self.config.feature_dim,
            dropout=getattr(self.config, 'dropout', 0.1),
            input_size=self.config.image_size
        )
        self.layout_pos_embed = nn.Embedding(self.layout_prompt_length, self.config.feature_dim)
        self.prompt_layer_norm = nn.LayerNorm(self.config.feature_dim)
        
        # Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=self.config.feature_dim,
            temperature=getattr(self.config, 'temperature', 0.1),
        )
        self.row_span_proj = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.feature_dim * 2, self.config.feature_dim),
            nn.LayerNorm(self.config.feature_dim)
        )
        self.col_span_proj = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.feature_dim * 2),
            nn.GELU(),
            nn.Linear(self.config.feature_dim * 2, self.config.feature_dim),
            nn.LayerNorm(self.config.feature_dim)
        )
        
        
    def prepare_layout_prompt(
        self,
        layout_embedding: torch.Tensor, 
    ) -> torch.Tensor:
        B, N, D = layout_embedding.shape
        
        # 1. Position embedding
        position_ids = torch.arange(N, device=layout_embedding.device)
        position_embeddings = self.layout_pos_embed(position_ids)
        
        # 2. Combine embeddings
        layout_prompt = layout_embedding + position_embeddings.unsqueeze(0)
        
        # 3. Final normalization
        layout_prompt = self.prompt_layer_norm(layout_prompt)
        
        return layout_prompt
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 이미지 인코딩
        encoder_outputs = self.image_encoder(batch['images'])
        visual_features = encoder_outputs.last_hidden_state
        visual_features = self.visual_proj(visual_features)
        visual_features = F.normalize(visual_features, p=2, dim=-1)

        # 레이아웃 인코딩
        layout_embedding = self.layout_encoder(batch['bboxes'], visual_features)
        layout_prompt = self.prepare_layout_prompt(layout_embedding)
        
        # 모드에 따라 다른 forward path 실행
        if self.training:
            return self._train_forward(batch, visual_features, layout_prompt)
        else:
            return self._eval_forward(batch, visual_features, layout_prompt)
    
    def _train_forward(self, batch, visual_features, layout_prompt):
        labels = batch['token_ids']     
        decoder_input_ids = shift_tokens_right(
            labels,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id
        )
        
        token_embeds = self.bart.model.decoder.embed_tokens(decoder_input_ids)
        prompt_inputs = torch.cat([layout_prompt, token_embeds], dim=1)
        
        # 1. Padding mask 생성
        padding_mask = torch.ones(prompt_inputs.size(0), prompt_inputs.size(1), device=prompt_inputs.device)
        padding_mask[:, layout_prompt.size(1):] = (decoder_input_ids != self.tokenizer.pad_token_id)
        
        decoder_outputs = self.bart.model.decoder(
            inputs_embeds=prompt_inputs,
            attention_mask=padding_mask,
            encoder_hidden_states=visual_features,
            encoder_attention_mask=torch.ones_like(padding_mask[:, :layout_prompt.size(1)]),
            use_cache=False,
            return_dict=True
        )
        
        last_hidden_state = decoder_outputs.last_hidden_state
        bbox_embeddings = last_hidden_state[:, :layout_prompt.size(1), :]  # layout_prompt 부분만
        logical_structure_embeddings = last_hidden_state[:, layout_prompt.size(1):, :]  # BOS + sequence + EOS

        tag_logits = self.bart.lm_head(logical_structure_embeddings)
        
        pointer_logits, empty_pointer_logits = self.layout_pointer(
            box_features=bbox_embeddings,
            tag_features=logical_structure_embeddings
        )
        row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
            bbox_embeddings,
            batch['bboxes']
        )
    
        row_span_coef, col_span_coef = get_coef_matrix(
            labels,
            self.tokenizer, 
            batch['html'],
            self.layout_prompt_length,
        )
        
        return {
            'tag_logits': tag_logits,
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_pointer_logits,
            'row_sim_matrix': row_sim_matrix,
            'col_sim_matrix': col_sim_matrix,
            'row_span_coef': row_span_coef,
            'col_span_coef': col_span_coef,
        }
        
    def _eval_forward(self, batch, visual_features, layout_prompt):
        B = layout_prompt.size(0)
        max_length = self.tokenizer.otsl_sequence_length
        prompt_length = layout_prompt.size(1)
        
        # 1. Layout prompt 초기 계산
        attention_mask = torch.ones((B, prompt_length), dtype=torch.bool, device=layout_prompt.device)
        init_outputs = self.bart.model.decoder(
            inputs_embeds=layout_prompt,
            attention_mask=attention_mask,
            encoder_hidden_states=visual_features,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 2. 토큰 생성 시작
        curr_ids = torch.full((B, 1), self.tokenizer.bos_token_id, dtype=torch.long, device=layout_prompt.device)
        past_key_values = init_outputs.past_key_values
        
        # 각 스텝의 hidden states를 저장할 리스트
        all_hidden_states = []
        
        # 생성 완료 여부를 추적하는 마스크
        unfinished_sequences = torch.ones(B, dtype=torch.bool, device=layout_prompt.device)
        
        # 토큰 생성 루프
        generated_tokens = []
        for step in range(max_length):
            curr_length = curr_ids.size(1)
            attention_mask = torch.ones((B, prompt_length + curr_length), dtype=torch.bool, device=layout_prompt.device)
            
            token_embeds = self.bart.model.decoder.embed_tokens(curr_ids[:, -1:])
            
            decoder_outputs = self.bart.model.decoder(
                inputs_embeds=token_embeds,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_features,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 현재 스텝의 hidden state 저장
            all_hidden_states.append(decoder_outputs.last_hidden_state)
            past_key_values = decoder_outputs.past_key_values
            
            logits = self.bart.lm_head(decoder_outputs.last_hidden_state)
            
            # EOS 토큰에 대한 bias 추가 (deterministic하게 유지)
            logits[:, :, self.tokenizer.eos_token_id] += 1.4
            
            # PAD 토큰은 생성하지 않도록 마스킹
            logits[:, :, self.tokenizer.pad_token_id] = float('-inf')
            
            # Temperature sampling 대신 argmax 사용 (deterministic)
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)
            
            # 완료되지 않은 시퀀스에 대해서만 토큰 추가
            next_token = next_token * unfinished_sequences.unsqueeze(1) + \
                        self.tokenizer.pad_token_id * (~unfinished_sequences.unsqueeze(1))
            
            generated_tokens.append(next_token)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # EOS 토큰이 생성된 시퀀스 체크
            unfinished_sequences = unfinished_sequences & (next_token != self.tokenizer.eos_token_id).squeeze(1)
            
            # 모든 시퀀스가 완료되었으면 중단
            if not unfinished_sequences.any():
                break
        
        # 생성된 토큰들을 하나의 텐서로 결합
        generated_sequence = torch.cat(generated_tokens, dim=1)  # (B, curr_length)
        # print(generated_sequence)
        # max_length까지 PAD 토큰으로 패딩
        if generated_sequence.size(1) < max_length:
            padding_length = max_length - generated_sequence.size(1)
            padding = torch.full((B, padding_length), self.tokenizer.pad_token_id, 
                               dtype=torch.long, device=generated_sequence.device)
            generated_sequence = torch.cat([generated_sequence, padding], dim=1)
        
        # hidden states도 패딩
        if len(all_hidden_states) < max_length:
            last_hidden = all_hidden_states[-1]
            padding_hidden = torch.zeros_like(last_hidden).repeat(1, max_length - len(all_hidden_states), 1)
            all_hidden_states.append(padding_hidden)
        
        # 3. 최종 hidden states 구성
        bbox_embeddings = init_outputs.last_hidden_state
        logical_structure_embeddings = torch.cat(all_hidden_states, dim=1)
        
        tag_logits = self.bart.lm_head(logical_structure_embeddings)
        
        pointer_logits, empty_pointer_logits = self.layout_pointer(
            box_features=bbox_embeddings,
            tag_features=logical_structure_embeddings
        )
        
        return {
            'tag_logits': tag_logits,
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_pointer_logits,
            'row_sim_matrix': None,
            'col_sim_matrix': None,
            'row_span_coef': None,
            'col_span_coef': None,
        }

    def get_sim_matrix(self, box_features: torch.Tensor, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = box_features.shape
        
        # bbox padding mask 생성 (모든 좌표가 0인 경우가 padding)
        bbox_mask = (bboxes.sum(dim=-1) != 0)  # (B, N)
        
        # 1. Project layout embeddings
        row_projected = self.row_span_proj(box_features)
        col_projected = self.col_span_proj(box_features)
        
        # 2. L2 normalize
        row_projected = F.normalize(row_projected, p=2, dim=-1)
        col_projected = F.normalize(col_projected, p=2, dim=-1)
        
        # 3. Compute pairwise similarities
        row_sim_matrix = torch.matmul(row_projected, row_projected.transpose(-2, -1))
        col_sim_matrix = torch.matmul(col_projected, col_projected.transpose(-2, -1))
        
        # 4. Create masks
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=box_features.device).unsqueeze(0)
        sim_padding_mask = bbox_mask.unsqueeze(-1) * bbox_mask.unsqueeze(-2)  # (B, N, N)
        final_mask = diag_mask & sim_padding_mask
        
        # 5. Apply masks
        row_sim_matrix = row_sim_matrix * final_mask
        col_sim_matrix = col_sim_matrix * final_mask
        
        return row_sim_matrix, col_sim_matrix
