#
#  Copyright (c) 2023 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from typing import Optional, Tuple, Union

import torch
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast

from autocontrastive_gen.modeling.multi_exit_modeling import MultiExitMixin


class MultiExitGPTNeo(GPTNeoForCausalLM, MultiExitMixin):
    def __init__(self, config, **multi_exit_kwargs):
        super().__init__(config)
        self.initialize_multi_exit_model(config,
                                         output_size=config.hidden_size,
                                         num_layers=config.num_hidden_layers,
                                         **multi_exit_kwargs)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """
        Based on the original forward() method of GPTNeoForCausalLM (transformers v4.26), with adaptations for
        multi-head loss and using self.contrast_function to calculate the logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        original_exit_output = self.lm_head.to(self.transformer.device)(hidden_states) if self.apply_lm_head \
            else hidden_states            

        index_to_layer_lm_logits = \
            self.calculate_all_layer_outputs(original_exit_output=original_exit_output,
                                             all_hidden_states=transformer_outputs.hidden_states,
                                             device=self.transformer.device)

        # TODO loss for conv_matrices
        # loss calculation for all the lm heads
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = 0
            for lm_logits in index_to_layer_lm_logits.values():

                # Compute loss in fp32 to match with mesh-tf version
                # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
                lm_logits = lm_logits.to(torch.float32)

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                layer_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss += layer_loss

            loss = loss.to(hidden_states.dtype)

        output_logits = self.calculate_model_output(original_exit_output=original_exit_output,
                                                    index_to_layer_lm_logits=index_to_layer_lm_logits)

        if not return_dict:
            output = (output_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=output_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def normalize_layer_output(self, layer_output, is_last_layer: bool):
        if not is_last_layer:  # layer norm for top layer is already applied within the GPT Neo code
            layer_output = self.transformer.ln_f(layer_output)
        return layer_output

