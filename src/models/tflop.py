from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import (
    SwinModel,
    SwinConfig,
    BartModel,
    BartConfig,
    AutoImageProcessor
)
from transformers.models.bart.modeling_bart import shift_tokens_right

from .layout_encoder import LayoutEncoder
from .layout_pointer import LayoutPointer
from .otsl_tokenizer import OTSLTokenizer
from utils.util import get_coef_matrix

class TFLOP(nn.Module):
    def __init__(self, config: Any, inference_mode: bool = False) -> None:
        super().__init__()
        self.config = config
        self.inference_mode = inference_mode
        # Tokenizer 초기화
        self.tokenizer = OTSLTokenizer(
            otsl_sequence_length=config.otsl_max_length
        )
        
        # tokenizer에 종속되게 설정        
        self.layout_prompt_length = config.total_sequence_length - config.otsl_max_length
        
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
        self.output_projection = nn.Linear(config.feature_dim, self.tokenizer.vocab_size, bias=True)

        # 4. Layout Pointer
        self.layout_pointer = LayoutPointer(
            feature_dim=config.feature_dim,
            temperature=getattr(config, 'temperature', 0.1),
        )
        self.row_span_proj = nn.Linear(config.feature_dim, config.feature_dim)
        self.col_span_proj = nn.Linear(config.feature_dim, config.feature_dim)

        # Layout prompt 관련 모듈들 추가
        self.rel_pos_embed = nn.Embedding(32, config.feature_dim)  # relative position embedding
        self.structure_q = nn.Linear(config.feature_dim, config.feature_dim)
        self.structure_k = nn.Linear(config.feature_dim, config.feature_dim)
        self.structure_v = nn.Linear(config.feature_dim, config.feature_dim)
        self.structure_ffn = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim * 4),
            nn.ReLU(),
            nn.Linear(config.feature_dim * 4, config.feature_dim)
        )

    def prepare_layout_prompt(
        self,
        layout_embedding: torch.Tensor,  # (B, N, D)
    ) -> torch.Tensor:
        """Layout embedding을 prompt로 변환

        Args:
            layout_embedding: Layout encoder의 출력 (B, N, D)
        """
        B, N, D = layout_embedding.shape

        # 1. Position embedding
        position_ids = torch.arange(N, device=layout_embedding.device)
        position_embeddings = self.layout_pos_embed(position_ids)  # (N, D)

        # 2. Structure-aware position embedding
        # 각 bbox의 상대적 위치 관계를 인코딩
        rel_pos_embeddings = self.get_relative_position_embeddings(N, D, device=layout_embedding.device)  # (N, N, D)
        
        # 3. Structure embedding
        # 각 bbox의 구조적 특성 강화
        structure_enhanced = self.structure_encoder(
            layout_embedding,  # (B, N, D)
            rel_pos_embeddings  # (N, N, D)
        )  # (B, N, D)

        # 4. Combine all embeddings
        layout_prompt = (
            structure_enhanced +  # 구조 정보
            position_embeddings.unsqueeze(0) +  # 절대 위치
            layout_embedding  # 원본 layout 정보
        )

        # 5. Final layer norm
        layout_prompt = self.prompt_layer_norm(layout_prompt)

        return layout_prompt

    def get_relative_position_embeddings(self, N: int, D: int, device: torch.device) -> torch.Tensor:
        """상대적 위치 관계를 인코딩"""
        pos_ids = torch.arange(N, device=device)
        rel_pos = pos_ids.unsqueeze(0) - pos_ids.unsqueeze(1)  # (N, N)
        
        # Relative position embedding table
        num_buckets = 32
        rel_pos = self.relative_position_bucket(rel_pos, num_buckets=num_buckets)
        rel_pos_embeddings = self.rel_pos_embed(rel_pos)  # (N, N, D)
        
        return rel_pos_embeddings

    def structure_encoder(self, layout_embedding: torch.Tensor, rel_pos_embeddings: torch.Tensor) -> torch.Tensor:
        """Layout의 구조적 특성을 강화"""
        B, N, D = layout_embedding.shape
        
        # Self-attention with relative position
        q = self.structure_q(layout_embedding)  # (B, N, D)
        k = self.structure_k(layout_embedding)  # (B, N, D)
        v = self.structure_v(layout_embedding)  # (B, N, D)
        
        # Attention scores with relative position bias
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, N, N)
        
        # Relative position bias - shape 맞추기
        rel_pos_bias = torch.matmul(
            rel_pos_embeddings,  # (N, N, D)
            k.mean(dim=0).transpose(-2, -1)  # (D, N)
        )  # (N, N, N)
        rel_pos_bias = rel_pos_bias.mean(dim=-1)  # (N, N)
        
        # Add relative position bias
        attn = attn + rel_pos_bias.unsqueeze(0)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # Combine with values
        out = torch.matmul(attn, v)  # (B, N, D)
        
        # FFN
        out = self.structure_ffn(out)  # (B, N, D)
        
        return out
    
    def prepare_cross_attention_mask(self, visual_features: torch.Tensor, layout_prompt: torch.Tensor) -> torch.Tensor:
        """Generate structured cross-attention mask
        
        Args:
            visual_features: (B, N, D)
            layout_prompt: (B, L, D)
        Returns:
            cross_attn_mask: (B, N) - boolean mask
        """
        B, N, D = visual_features.shape
        
        # 1. Layout-aware attention mask
        layout_attn = torch.matmul(
            layout_prompt,  # (B, L, D)
            visual_features.transpose(-2, -1)  # (B, D, N)
        ) / math.sqrt(D)  # (B, L, N)
        
        # 2. Convert to boolean mask
        # Mean over layout dimension to get importance per visual feature
        feature_importance = layout_attn.mean(dim=1)  # (B, N)
        
        # Convert to boolean mask - keep features above mean importance
        mean_importance = feature_importance.mean(dim=1, keepdim=True)
        std_importance = feature_importance.std(dim=1, keepdim=True)
        threshold = mean_importance - 0.5 * std_importance  # mean - 0.5*std as threshold
        
        cross_attn_mask = (feature_importance > threshold).to(torch.bool)  # (B, N)
        
        # Ensure at least half of the features are attended to
        min_features = N // 2
        for b in range(B):
            if cross_attn_mask[b].sum() < min_features:
                # If too few features are selected, take top k by value
                _, top_indices = torch.topk(feature_importance[b], min_features)
                cross_attn_mask[b] = torch.zeros_like(cross_attn_mask[b])
                cross_attn_mask[b][top_indices] = True
        
        return cross_attn_mask

    def enhance_visual_features(self, visual_features: torch.Tensor, layout_prompt: torch.Tensor) -> torch.Tensor:
        """Enhance visual features with layout information
        
        Args:
            visual_features: (B, N, D)
            layout_prompt: (B, L, D)
        Returns:
            enhanced_features: (B, N, D)
        """
        # Layout-guided attention
        attn_weights = torch.matmul(
            visual_features,  # (B, N, D)
            layout_prompt.transpose(-2, -1)  # (B, D, L)
        ) / math.sqrt(visual_features.size(-1))  # (B, N, L)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Combine with original features
        enhanced_features = visual_features + torch.matmul(attn_weights, layout_prompt)
        enhanced_features = F.layer_norm(enhanced_features, normalized_shape=[enhanced_features.size(-1)])
        
        return enhanced_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing:
                - images: (B, C, H, W)
                - bboxes: (B, num_boxes, 4)
                - data_tag_mask: (B, total_length) - non-empty C 태그 위치
                - empty_tag_mask: (B, total_length) - empty C 태그 위치
                ...
        """
        images = batch['images']           
        text_regions = batch['bboxes']  
        data_tag_mask = batch['data_tag_mask']  # non-empty C 태그
        empty_tag_mask = batch['empty_tag_mask']  # empty C 태그
        
        B = len(images)
        
        # 1. Image Encoding
        encoder_outputs = self.image_encoder(images, output_hidden_states=True)
        visual_features = encoder_outputs.last_hidden_state
        visual_features = F.normalize(visual_features, p=2, dim=-1)
        visual_features = self.visual_proj(visual_features)

        # 2. Layout encoding & prompt preparation
        layout_embedding = self.layout_encoder(text_regions, visual_features)
        layout_prompt = self.prepare_layout_prompt(layout_embedding)
        
        # 3. Enhance visual features with layout information
        enhanced_visual_features = self.enhance_visual_features(visual_features, layout_prompt)
        
        # 4. Prepare cross-attention mask
        cross_attn_mask = self.prepare_cross_attention_mask(enhanced_visual_features, layout_prompt)

        if self.training:
            labels = batch['token_ids']     
            attention_mask = batch['attention_mask']   # (B, total_sequence_length)
            
            
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
            
            decoder_outputs = self.bart.decoder(
                inputs_embeds=prompt_inputs,
                attention_mask=attention_mask,
                encoder_hidden_states=enhanced_visual_features,  # 강화된 visual features 사용
                encoder_attention_mask=cross_attn_mask,  # 구조적 cross-attention mask 사용
                use_cache=False,
                output_hidden_states=True,
                return_dict=True
            )

        else :
            # 초기 BOS 토큰으로 시작
            curr_ids = torch.full(
                (B, 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
                device=images.device
            )
            max_length = self.tokenizer.otsl_sequence_length
            
            # Layout prompt의 길이
            prompt_length = layout_prompt.size(1)
            
            for step in range(max_length-1):
                # 현재까지의 시퀀스 길이
                curr_length = curr_ids.size(1)
                
                # Attention mask 생성:
                # - layout prompt 부분은 모두 True
                # - 현재까지 생성된 토큰들은 True
                # - 아직 생성되지 않은 부분은 False
                attention_mask = torch.zeros(
                    (B, prompt_length + max_length),
                    dtype=torch.bool,
                    device=images.device
                )
                attention_mask[:, :prompt_length] = True  # layout prompt 부분
                attention_mask[:, prompt_length:prompt_length + curr_length] = True  # 생성된 토큰들
                
                token_embeds = self.bart.decoder.embed_tokens(curr_ids)
                prompt_inputs = torch.cat([layout_prompt, token_embeds], dim=1)
                
                decoder_outputs = self.bart.decoder(
                    inputs_embeds=prompt_inputs,
                    attention_mask=attention_mask[:, :prompt_length + curr_length],
                    encoder_hidden_states=enhanced_visual_features,
                    encoder_attention_mask=cross_attn_mask,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                last_hidden_state = decoder_outputs.last_hidden_state
                logits = self.output_projection(last_hidden_state[:, -1:])
                
                # Nucleus sampling with constraints
                next_token = self.constrained_nucleus_sampling(
                    logits.squeeze(1),
                    curr_ids,
                    p=0.9,
                    temperature=0.7
                )
                
                
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # training과 동일한 입력 형태 구성
            decoder_input_ids = curr_ids
            token_embeds = self.bart.decoder.embed_tokens(decoder_input_ids)
            prompt_inputs = torch.cat([
                layout_prompt,
                token_embeds
            ], dim=1)
            
            decoder_outputs = self.bart.decoder(
                inputs_embeds=prompt_inputs,
                encoder_hidden_states=enhanced_visual_features,
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
            tag_features=logical_structure_embeddings,
            data_tag_mask = data_tag_mask,
        )

        row_sim_matrix, col_sim_matrix = None, None
        row_span_coef, col_span_coef = None, None
        if self.training:
            row_sim_matrix, col_sim_matrix = self.get_sim_matrix(
                bbox_embeddings,
                attention_mask=attention_mask[:, :self.layout_prompt_length]
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

    def get_sim_matrix(self, box_features: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # 4. Apply attention mask
        mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # (B, N, N)
        row_sim_matrix = row_sim_matrix * mask
        col_sim_matrix = col_sim_matrix * mask
        
        # 5. Set diagonal to zero to exclude self-similarity
        diag_mask = ~torch.eye(N, device=box_features.device, dtype=torch.bool).unsqueeze(0)
        row_sim_matrix = row_sim_matrix * diag_mask
        col_sim_matrix = col_sim_matrix * diag_mask
        
        return row_sim_matrix, col_sim_matrix

    def relative_position_bucket(self, relative_position: torch.Tensor, num_buckets: int = 32) -> torch.Tensor:
        """상대적 위치를 bucket index로 변환"""
        # 1. 절대값 계산
        relative_buckets = torch.abs(relative_position)
        
        # 2. Sign 정보 보존 (음수: 0, 양수: num_buckets/2)
        relative_buckets = torch.where(
            relative_position < 0,
            relative_buckets,
            relative_buckets + num_buckets // 2
        )
        
        # 3. Bucketing
        max_distance = num_buckets // 2
        relative_buckets = torch.clamp(relative_buckets, 0, max_distance - 1)
        
        return relative_buckets
    
    def constrained_nucleus_sampling(
        self,
        logits: torch.Tensor,  # (B, vocab_size)
        current_tokens: torch.Tensor,  # (B, curr_length)
        p: float = 0.9,
        temperature: float = 0.7
    ) -> torch.Tensor:
        """OTSL 문법 규칙을 고려한 Nucleus (top-p) sampling

        Args:
            logits: 토큰 예측을 위한 logits
            current_tokens: 현재까지 생성된 토큰들
            p: nucleus sampling의 확률 임계값
            temperature: 샘플링 temperature
        """
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
        
        for b in range(current_tokens.size(0)):
            curr_seq = current_tokens[b]
            curr_row_idx = (curr_seq == self.tokenizer.nl_token_id).sum()
            curr_col_idx = curr_seq.size(0) - curr_seq.eq(self.tokenizer.nl_token_id).sum() - 1
            
            # 모든 제약 조건 적용 전에 최소한 C 토큰은 항상 가능하도록 설정
            valid_mask[b, self.tokenizer.c_token_id] = True
            
            # 1. 마지막 토큰이 NL이어야 함 (max_length-2 위치에서)
            if curr_seq.size(0) == self.tokenizer.otsl_sequence_length - 1:
                valid_mask[b, :] = False
                valid_mask[b, self.tokenizer.nl_token_id] = True
                continue  # 다른 제약조건 확인 불필요
            
            # 2. 연속된 NL 방지
            if curr_seq[-1] == self.tokenizer.nl_token_id:
                valid_mask[b, self.tokenizer.nl_token_id] = False
            
            # 3. 최소 2개의 real column 보장
            if curr_col_idx == 0:  # 첫 번째 컬럼일 때
                valid_mask[b, self.tokenizer.nl_token_id] = False  # NL 방지
            
            # 4. 최소 2개의 row 보장
            if curr_row_idx == 0:  # 첫 번째 행일 때
                if curr_col_idx < 1:  # 최소 2개의 컬럼이 없으면
                    valid_mask[b, self.tokenizer.nl_token_id] = False  # NL 방지

            # OTSL 문법 규칙
            # 첫 번째 행은 L과 C만 허용
            if curr_row_idx == 0:
                valid_mask[b, self.tokenizer.u_token_id] = False
                valid_mask[b, self.tokenizer.x_token_id] = False
            
            # 첫 번째 열은 U와 C만 허용
            if curr_col_idx == 0:
                valid_mask[b, self.tokenizer.l_token_id] = False
                valid_mask[b, self.tokenizer.x_token_id] = False
                
            # L 토큰의 왼쪽은 L 또는 C여야 함
            if curr_col_idx > 0 and curr_seq[-1] == self.tokenizer.l_token_id:
                if curr_seq[-2] not in [self.tokenizer.c_token_id, self.tokenizer.l_token_id]:
                    valid_mask[b, self.tokenizer.l_token_id] = False
                    
            # U 토큰의 위쪽은 U 또는 C여야 함
            if curr_row_idx > 0:
                prev_row_tokens = current_tokens[b][:-curr_col_idx-1]
                if prev_row_tokens[-1] not in [self.tokenizer.c_token_id, self.tokenizer.u_token_id]:
                    valid_mask[b, self.tokenizer.u_token_id] = False

            # X 토큰 규칙
            if curr_row_idx > 0 and curr_col_idx > 0:
                left_token = curr_seq[-1]
                if left_token not in [self.tokenizer.x_token_id, self.tokenizer.u_token_id]:
                    valid_mask[b, self.tokenizer.x_token_id] = False
                
                up_token = current_tokens[b][:-curr_col_idx-1][-1]
                if up_token not in [self.tokenizer.x_token_id, self.tokenizer.l_token_id]:
                    valid_mask[b, self.tokenizer.x_token_id] = False

            # 모든 토큰이 마스킹된 경우 C 토큰은 허용
            if not valid_mask[b].any():
                valid_mask[b, self.tokenizer.c_token_id] = True

        # Temperature scaling
        logits = logits / temperature
        
        # 마스킹 적용
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        
        # Nucleus sampling
        probs = F.softmax(logits, dim=-1)
        
        # 수치 안정성을 위한 확인
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # fallback: 유효한 토큰 중에서 균등하게 선택
            probs = valid_mask.float()
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # cumsum 대신 직접 계산
        cumulative_probs = torch.zeros_like(sorted_probs)
        for i in range(sorted_probs.size(-1)):
            if i == 0:
                cumulative_probs[..., i] = sorted_probs[..., i]
            else:
                cumulative_probs[..., i] = cumulative_probs[..., i-1] + sorted_probs[..., i]
        
        # 임계값 p를 넘는 확률 마스킹
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 마스킹 적용 및 재정규화
        sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # 5. 샘플링
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
        next_tokens = torch.gather(sorted_indices, -1, sampled_indices)
        
        return next_tokens
