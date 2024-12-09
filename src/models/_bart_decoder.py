import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartDecoder

class TFLOPDecoder(BartDecoder):
    """Custom BART Decoder for TFLOP
    
    Layout prompt와 visual features를 cross-attention으로 처리하는 decoder
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        layout_prompt=None,
        visual_features=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Combine visual features and layout prompt for cross-attention
        if visual_features is not None and layout_prompt is not None:
            cross_states = torch.cat([visual_features, layout_prompt], dim=1)
            
            # Create cross attention mask
            if encoder_attention_mask is not None:
                layout_attention_mask = torch.ones(
                    (layout_prompt.size(0), layout_prompt.size(1)),
                    device=encoder_attention_mask.device
                )
                cross_attention_mask = torch.cat(
                    [encoder_attention_mask, layout_attention_mask], 
                    dim=1
                )
            else:
                cross_attention_mask = None
        else:
            cross_states = visual_features
            cross_attention_mask = encoder_attention_mask

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=cross_states,
            encoder_attention_mask=cross_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 