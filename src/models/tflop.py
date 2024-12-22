from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartForCausalLM, BartConfig
from transformers.file_utils import ModelOutput
from .custom_swin_encoder import CustomSwinEncoder
from .custom_bart_decoder import CustomBartDecoder
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
        self.image_encoder = CustomSwinEncoder(
            input_size = self.config.image_size,
            align_long_axis = False,
            window_size = 8,
            encoder_layer = [2, 2, 14, 2]
        )
        self.visual_proj = nn.Linear(
            self.image_encoder.feature_dim,
            self.config.feature_dim
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
            
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # decoder_start_token_id = self.tokenizer.bos_token_id,
        )
        
        self.logical_structure_decoder = CustomBartDecoder(self.bart_config)
        
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
        self.row_span_proj = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        self.col_span_proj = nn.Linear(self.config.feature_dim, self.config.feature_dim)
        
        
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
        image_tensors = self.image_encoder.prepare_input(batch['images'])
        visual_features = self.image_encoder(image_tensors)
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
        """Teacher forcing: Use ground truth tags for training"""
        # Prepare inputs: [layout_prompt | target_tags]
        labels = batch['token_ids']
        decoder_inputs = layout_prompt
        
        # Generate with teacher forcing
        outputs = self.logical_structure_decoder(
            input_ids = decoder_inputs, 
            encoder_hidden_states = visual_features,
            labels = labels,
        )
        
        # Extract features for alignment
        hidden_states = outputs.hidden_states[-1]
        box_features = hidden_states[:, :layout_prompt.size(1), :]
        tag_features = hidden_states[:, layout_prompt.size(1):, :]
        
        # Compute box-tag alignments
        pointer_logits, empty_logits = self.layout_pointer(box_features, tag_features)
        row_sim, col_sim = self.get_sim_matrix(box_features)
        row_coef, col_coef = get_coef_matrix(target_tags, self.tokenizer, self.layout_prompt_length)
        
        return {
            'tag_logits': outputs.logits[:, self.layout_prompt_length:, :],
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_logits,
            'row_sim_matrix': row_sim,
            'col_sim_matrix': col_sim,
            'row_span_coef': row_coef,
            'col_span_coef': col_coef,
        }
        
    def _eval_forward(self, batch, visual_features, layout_prompt):
        batch_size = layout_prompt.size(0)
        
        # 초기 입력 설정
        curr_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=layout_prompt.device
        )
        
        prompt_length = layout_prompt.size(1)

        # Greedy decoding
        for step in range(self.tokenizer.otsl_sequence_length - 1):
            curr_length = curr_ids.size(1)
            attention_mask = torch.ones(
                    (batch_size, prompt_length + curr_length),
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

        decoder_input_ids = curr_ids
        token_embeds = self.bart.model.decoder.embed_tokens(decoder_input_ids)
        prompt_inputs = torch.cat([
            layout_prompt,
            token_embeds
        ], dim=1)
        
        decoder_outputs = self.bart.model.decoder(
            inputs_embeds=prompt_inputs,
            encoder_hidden_states=visual_features,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        # 마지막 hidden states 추출
        last_hidden_state = decoder_outputs.hidden_states[-1]
        bbox_embeddings = last_hidden_state[:, :layout_prompt.size(1), :]
        logical_structure_embeddings = last_hidden_state[:, layout_prompt.size(1):, :]
        
        pointer_logits, empty_pointer_logits = self.layout_pointer(
            box_features=bbox_embeddings,
            tag_features=logical_structure_embeddings
        )
        
        return {
            'tag_logits': outputs.logits[:, layout_prompt.size(1):, :],
            'pointer_logits': pointer_logits,
            'empty_pointer_logits': empty_pointer_logits,
            'generated_ids': generated_ids,
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
