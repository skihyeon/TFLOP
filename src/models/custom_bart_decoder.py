import torch
import torch.nn as nn
from transformers import MBartForCausalLM, MBartConfig
from typing import Optional
from transformers.file_utils import ModelOutput


class MyBartDecoder(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        
        self.model = MBartForCausalLM(
            config= MBartConfig(
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
                activation_function='gelu',
                dropout=0.1,
                attention_dropout=0.1,
            )
        )
        self.model.forward = self.forward
        self.model.config.is_encoder_decoder = True
        # self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        
    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        
    def forward(self,
                input_ids,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = True,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[torch.Tensor] = None,
                return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        
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
        
        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )