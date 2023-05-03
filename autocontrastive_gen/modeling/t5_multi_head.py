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
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class MultiHeadT5(T5ForConditionalGeneration):
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
            {self.lm_head_name_prefix+str(idx): nn.Linear(config.d_model, config.vocab_size, bias=False)
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
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        Based on the original forward() method of T5ForConditionalGeneration (transformers v4.26), with adaptations for
        multi-head loss and using self.contrast_function to calculate the logits
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        def normalize_and_rescale(hidden_layer, is_top_layer):
            if not is_top_layer:  # layer norm and dropout for top layer are already applied within the T5 decoder
                normalized_hidden_layer = self.decoder.final_layer_norm(hidden_layer)
                hidden_layer = self.decoder.dropout(normalized_hidden_layer)
            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                hidden_layer = hidden_layer * (self.model_dim ** -0.5)
            return hidden_layer

        # index 0 of decoder_outputs.hidden_states are the decoder input embeddings, index 1 is the 1st layer,
        # index 2 is the 2nd, etc.
        index_to_layer_lm_head_logits = {}
        for head_name, layer_lm_head in self.name_to_lm_exit_head.items():
            layer_index = int(head_name.split(self.lm_head_name_prefix)[-1])
            is_top_layer = layer_index == self.config.num_decoder_layers
            layer_decoder_outputs = normalize_and_rescale(decoder_outputs.hidden_states[layer_index], is_top_layer)
            layer_lm_logits = layer_lm_head.to(self.decoder.device)(layer_decoder_outputs)
            index_to_layer_lm_head_logits[layer_index] = layer_lm_logits

        # loss calculation for all the lm heads
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = 0
            for idx, lm_logits in index_to_layer_lm_head_logits.items():
                layer_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss += layer_loss

        # according to the initialization, we decide which head(s) are used for generating outputs at inference time
        if self.use_original_head:
            output_logits = self.lm_head(normalize_and_rescale(sequence_output, True))
        elif not self.contrast_layer_indices:
            output_logits = index_to_layer_lm_head_logits[self.output_layer_index]
        else:
            lm_logits_upper = self.lm_head(normalize_and_rescale(sequence_output, True)) if self.contrast_layer_indices[0] == 'original' \
                else index_to_layer_lm_head_logits[self.contrast_layer_indices[0]]
            lm_logits_lower = index_to_layer_lm_head_logits[self.contrast_layer_indices[1]]

            output_logits = self.contrast_function(lm_logits_upper, lm_logits_lower)

        if not return_dict:
            output = (output_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=output_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
