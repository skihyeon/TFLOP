import torch
import torch.nn as nn
from transformers import BartForCausalLM, BartConfig
from typing import Optional
from transformers.file_utils import ModelOutput



class CustomBartDecoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.bart = BartForCausalLM(config)
        self.bart.forward = self.forward    # bartforcausallm이 generate 사용하려면 forward 수정해야함
        self.bart.config.is_encoder_decoder = True  # bartforcausallm의 encoder params는 로딩하지 않으면서 embedding은 넣어주기 위한 trick
        self.bart.model.decoder.embed_tokens.padding_idx = config.pad_token_id
        self.bart.prepare_inputs_for_generation = self.prepare_inputs_for_inference

    
    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, 
                                    encoder_outputs: torch.Tensor, 
                                    past_key_values=None, 
                                    past=None, 
                                    use_cache: bool = None, 
                                    attention_mask: torch.Tensor = None):
        if past: past_key_values = past
        attention_mask = input_ids.ne(self.config.pad_token_id).long()
        if past_key_values: input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = None,
                output_attentions: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[torch.Tensor] = None,
                return_dict: bool = None,
                ):
        output_attentions = output_attentions if output_attentions else self.bart.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states else self.bart.config.output_hidden_states
        return_dict = return_dict if return_dict else self.bart.config.use_return_dict

        outputs = self.model.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.bart.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss else output

        return ModelOutput(
            loss = loss,
            logits = logits,
            past_key_values = outputs.past_key_values,
            hidden_states = outputs.hidden_states,
            decoder_attentions = outputs.attentions,
            cross_attentions = outputs.cross_attentions,
        )