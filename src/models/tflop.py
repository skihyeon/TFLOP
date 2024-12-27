from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Swinv2Model, Swinv2Config, BartForCausalLM, BartConfig, BartModel
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
            # decoder_attention_heads=16, 
            
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.feature_dim,
            max_position_embeddings=self.config.total_sequence_length,
            
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            activation_function='gelu',
            dropout=0.1,
            attention_dropout=0.1,
        )
        
        self.bart = BartForCausalLM(self.bart_config)
        # self.bart = BartModel(self.bart_config)
        # self.bart.config.is_encoder_decoder = True
        self.bart.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.output_projection = nn.Linear(self.config.feature_dim, self.tokenizer.vocab_size)
    def other_setup(self):
        # Layout Encoder
        self.layout_encoder = LayoutEncoder(
            feature_dim=self.config.feature_dim,
            dropout=getattr(self.config, 'dropout', 0.1),
            input_size=self.config.image_size
        )
        self.layout_pos_embed = nn.Embedding(self.config.total_sequence_length, self.config.feature_dim)
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

        position_ids = torch.arange(N, device=layout_embedding.device)
        position_embeddings = self.layout_pos_embed(position_ids) 
        layout_prompt = layout_embedding + position_embeddings.unsqueeze(0)
        layout_prompt = self.prompt_layer_norm(layout_prompt)
        
        return layout_prompt
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 이미지 인코딩
        encoder_outputs = self.image_encoder(batch['images'], output_hidden_states=True)
        visual_features = encoder_outputs.last_hidden_state
        visual_features = F.normalize(visual_features, p=2, dim=-1)
        visual_features = self.visual_proj(visual_features)

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
        prompt_inputs = torch.cat([
            layout_prompt,
            token_embeds
        ], dim=1)
        
        # 1. Padding mask 생성
        padding_mask = torch.ones(prompt_inputs.size(0), prompt_inputs.size(1), device=prompt_inputs.device)
        padding_mask[:, layout_prompt.size(1):] = (decoder_input_ids != self.tokenizer.pad_token_id)
        
        decoder_outputs = self.bart.model.decoder(
            inputs_embeds=prompt_inputs,
            attention_mask=padding_mask,  # padding mask만 전달
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
        row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
            bbox_embeddings
        )
    
        row_span_coef, col_span_coef = get_coef_matrix(
            labels,
            self.tokenizer, 
            self.layout_prompt_length
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
        curr_ids = torch.full(
                (B, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=layout_prompt.device
            )
        max_length = self.tokenizer.otsl_sequence_length
        prompt_length = layout_prompt.size(1)

        # 토큰 생성 루프
        for step in range(max_length):
            curr_length = curr_ids.size(1)
            attention_mask = torch.ones(
                    (B, prompt_length + curr_length),
                    dtype=torch.bool,
                    device=layout_prompt.device
                )           
            token_embeds = self.bart.model.decoder.embed_tokens(curr_ids)
            prompt_inputs = torch.cat([layout_prompt, token_embeds], dim=1)          

            decoder_outputs = self.bart.model.decoder(
                inputs_embeds=prompt_inputs,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_features,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = decoder_outputs.last_hidden_state
            logits = self.output_projection(last_hidden_state[:, -1:])    
            next_token = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)    
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        # 마지막 step의 hidden states 사용
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
            'row_sim_matrix': None,
            'col_sim_matrix': None,
            'row_span_coef': None,
            'col_span_coef': None,
        }

    def get_sim_matrix(self, box_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = box_features.shape
        
        # 1. Project layout embeddings
        row_projected = self.row_span_proj(box_features)  # (B, N, D)
        col_projected = self.col_span_proj(box_features)  # (B, N, D)
        
        # 2. L2 normalize
        row_projected = F.normalize(row_projected, p=2, dim=-1)
        col_projected = F.normalize(col_projected, p=2, dim=-1)
        
        # 3. Compute pairwise similarities
        row_sim_matrix = torch.matmul(row_projected, row_projected.transpose(-2, -1))  # (B, N, N)
        col_sim_matrix = torch.matmul(col_projected, col_projected.transpose(-2, -1))  # (B, N, N)
        # 5. Set diagonal to zero to exclude self-similarity
        diag_mask = ~torch.eye(N, device=box_features.device, dtype=torch.bool).unsqueeze(0)
        row_sim_matrix = row_sim_matrix * diag_mask
        col_sim_matrix = col_sim_matrix * diag_mask
        
        return row_sim_matrix, col_sim_matrix
