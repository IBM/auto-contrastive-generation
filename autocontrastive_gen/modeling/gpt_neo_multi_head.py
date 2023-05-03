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
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast


class MultiHeadGPTNeo(GPTNeoForCausalLM):
    def __init__(self, config, use_original_head=False, output_layer_index=24, contrast_layer_indices=None,
                 contrast_function=None, freeze_parameters=True):
        super().__init__(config)
        if not hasattr(config, 'lm_head_layer_indices'):
            raise Exception(f"LM exit head indices must be specified when initializing {self.__class__.__name__}")

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False  # freeze all standard parameters and heads

        # initialize the linear exit heads
        self.lm_head_name_prefix = 'lm_head_'
        self.name_to_lm_exit_head = nn.ModuleDict(
            {self.lm_head_name_prefix+str(idx): nn.Linear(config.hidden_size, config.vocab_size, bias=False)
             for idx in config.lm_head_layer_indices})

        # set inference-time generation parameters
        self.use_original_head = use_original_head
        self.output_layer_index = output_layer_index
        self.contrast_layer_indices = contrast_layer_indices
        self.contrast_function = contrast_function
        desc = 'original head' if self.use_original_head else (
            f'output layer {self.output_layer_index}' if not self.contrast_layer_indices else f'contrast between layers {self.contrast_layer_indices}')
        print(f'********* Using {desc} for generation ***********')

        if freeze_parameters:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)

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

        # index 0 of transformer_outputs.hidden_states are the decoder input embeddings, index 1 is the 1st layer,
        # index 2 is the 2nd, etc.
        index_to_layer_lm_head_logits = {}
        for head_name, layer_lm_head in self.name_to_lm_exit_head.items():
            layer_index = int(head_name.split(self.lm_head_name_prefix)[-1])
            is_top_layer = layer_index == self.config.num_hidden_layers
            layer_outputs = transformer_outputs.hidden_states[layer_index]
            if not is_top_layer:  # layer norm for top layer is already applied within the GPT2 code
                layer_outputs = self.transformer.ln_f(layer_outputs)
            layer_lm_logits = layer_lm_head.to(self.transformer.device)(layer_outputs)
            index_to_layer_lm_head_logits[layer_index] = layer_lm_logits

        # loss calculation for all the lm heads
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = 0
            for idx, lm_logits in index_to_layer_lm_head_logits.items():

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

        # according to the initialization, we decide which head(s) are used for generating outputs at inference time
        if self.use_original_head:
            output_logits = self.lm_head(hidden_states)
        elif not self.contrast_layer_indices:
            output_logits = index_to_layer_lm_head_logits[self.output_layer_index]
        else:
            lm_logits_upper = self.lm_head(hidden_states) if self.contrast_layer_indices[0] == 'original' \
                else index_to_layer_lm_head_logits[self.contrast_layer_indices[0]]
            lm_logits_lower = index_to_layer_lm_head_logits[self.contrast_layer_indices[1]]

            contrasted = self.contrast_function(lm_logits_upper, lm_logits_lower)

            output_logits = contrasted

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
