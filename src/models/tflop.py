from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Swinv2Model, Swinv2Config, BartForCausalLM, BartConfig

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
        self.visual_proj = nn.Linear(
            self.image_encoder.config.hidden_size,
            self.config.feature_dim
        )
        
    def bart_setup(self):
        self.bart_config = BartConfig(
            is_decoder=True,
            is_encoder_decoder=False,
            add_cross_attention=True,
            decoder_layers=4,  # Donut의 설정만 가져오기
            
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.feature_dim,
            max_position_embeddings=self.config.total_sequence_length,
            
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        self.bart = BartForCausalLM(self.bart_config)
        self.bart.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        
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
        """Teacher forcing: Use ground truth tags for training"""
        # Prepare inputs: [layout_prompt | target_tags]
        target_tags = batch['token_ids']
        tag_embeds = self.bart.model.decoder.embed_tokens(target_tags)
        decoder_inputs = torch.cat([layout_prompt, tag_embeds], dim=1)
        
        # Prepare labels: [-100 padding | target_tags]
        label_padding = torch.full(
            (target_tags.size(0), layout_prompt.size(1)),
            fill_value=-100,
            dtype=target_tags.dtype,
            device=target_tags.device
        )
        labels = torch.cat([label_padding, target_tags], dim=1)
        
        # Generate with teacher forcing
        outputs = self.bart(
            inputs_embeds=decoder_inputs,
            encoder_hidden_states=visual_features,
            attention_mask=torch.ones(layout_prompt.size()[:2], device=layout_prompt.device),
            labels=labels,
            output_hidden_states=True,
            return_dict=True
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
        bos_ids = torch.full((batch_size, 1), self.tokenizer.bos_token_id, device=layout_prompt.device)
        bos_embeds = self.bart.model.decoder.embed_tokens(bos_ids)
        current_embeds = torch.cat([layout_prompt, bos_embeds], dim=1)
        
        generated_ids = []
        attention_mask = torch.ones(batch_size, current_embeds.size(1), device=layout_prompt.device)
        
        # 디버깅을 위한 정보
        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"BOS token id: {self.tokenizer.bos_token_id}")
        print(f"EOS token id: {self.tokenizer.eos_token_id}")
        
        # Greedy decoding
        for step in range(self.tokenizer.otsl_sequence_length):
            outputs = self.bart(
                inputs_embeds=current_embeds,
                encoder_hidden_states=visual_features,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            # 디버깅: logits 분포 확인
            if step < 3:  # 처음 3 스텝만 출력
                print(f"\nStep {step} logits statistics:")
                print(f"Logits shape: {next_token_logits.shape}")
                print(f"Logits mean: {next_token_logits.mean().item():.4f}")
                print(f"Logits std: {next_token_logits.std().item():.4f}")
                
                # Top-5 토큰과 확률 출력
                probs = torch.softmax(next_token_logits, dim=-1)
                top_probs, top_indices = probs[0].topk(5)
                print("Top 5 predictions:")
                for idx, prob in zip(top_indices, top_probs):
                    token = self.tokenizer.id2token.get(idx.item(), f"UNKNOWN_{idx.item()}")
                    print(f"Token: {token}, ID: {idx.item()}, Prob: {prob.item():.4f}")
            
            # 유효하지 않은 토큰 마스킹
            invalid_tokens = [self.tokenizer.pad_token_id]  # 필요한 경우 다른 토큰 추가
            for invalid_id in invalid_tokens:
                if invalid_id is not None:
                    next_token_logits[:, invalid_id] = float('-inf')
            
            next_token = next_token_logits.argmax(dim=-1)
            
            # 디버깅: 생성된 토큰 확인
            if step < 3:
                print(f"Generated token at step {step}: {next_token.tolist()}")
                print(f"Token text: {[self.tokenizer.id2token.get(t.item(), 'UNK') for t in next_token]}")
            
            # EOS 토큰 체크
            if (next_token == self.tokenizer.eos_token_id).any():
                print(f"EOS token generated at step {step}")
                break
            
            generated_ids.append(next_token)
            
            # 다음 스텝 준비
            next_embeds = self.bart.model.decoder.embed_tokens(next_token.unsqueeze(-1))
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, device=layout_prompt.device)
            ], dim=1)
        
        # 생성된 시퀀스 처리
        generated_ids = torch.stack(generated_ids, dim=1)  # (batch_size, seq_len)
        
        # 마지막 hidden states 추출
        last_hidden_state = outputs.hidden_states[-1]
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
