import functools
from abc import abstractmethod
from typing import Optional, Union, Callable

import torch
from torch import nn

from autocontrastive_gen.modeling.configuration import VocabularyProjectionMode


class MultiExitModel:
    """
    This class adds multi-exit functionality to a model with a language modeling head,
    i.e., add the capability to produce next-token predictions from one of the intermediate model layers
    or by contrasting the outputs of different layers.

    A multi-exit model class should inherit from both this class and the appropriate LM class in the transformers
    library, for example:
    `class MultiExitGPT2(MultiExitModel, GPT2LMHeadModel):`
    """
    vocab_projection_mode: VocabularyProjectionMode

    name_to_lm_exit_head: Optional[nn.ModuleDict]  # for VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION
    name_to_conv_matrix: Optional[nn.ModuleDict]  # for VocabularyProjectionMode.SHARED_PROJECTION_CAST_OUTPUTS

    lm_head_name_prefix = 'lm_head_'
    conv_matrix_name_suffix = Optional[str]

    use_original_head: bool
    output_layer_index: int
    contrast_layer_indices: Optional[tuple[Union[int, str], int]]
    contrast_function: Optional[Callable]

    freeze_parameters: bool

    apply_lm_head = True  # the LM projection can be turned off, e.g., for learning layer-to-layer conversions

    def __init__(self, config,
                 vocab_projection_mode: VocabularyProjectionMode,
                 use_original_head: bool,
                 output_layer_index: int,
                 contrast_layer_indices: Optional[tuple[Union[int, str], int]],
                 contrast_function: Optional[Callable],
                 freeze_parameters=False):
        if not hasattr(config, 'lm_head_layer_indices'):
            raise Exception(f"LM exit head indices must be specified when initializing {self.__class__.__name__}")

        super().__init__(config)

        # We use a dedicated wrapper - instead of just overriding *forward()* - to preserve the original signature (the
        # transformers library does some validations of expected *forward()* input arguments, which differ by model)
        def forward_method_wrapper(*args, **kwargs):
            return self._forward(*args, **kwargs)
        self.forward = functools.update_wrapper(wrapper=forward_method_wrapper, wrapped=self.forward)

        self.vocab_projection_mode = vocab_projection_mode
        self.output_size = getattr(config, self.output_size_config_key)
        self.num_layers = getattr(config, self.num_layers_config_key)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False  # freeze all standard parameters and heads

        if self.vocab_projection_mode == VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION:
            # initialize the linear exit heads
            self.name_to_lm_exit_head = nn.ModuleDict(
                {self.lm_head_name_prefix + str(layer): self.lm_head if isinstance(layer, str) and layer == 'original'
                    else nn.Linear(self.output_size, config.vocab_size, bias=False)
                 for layer in config.lm_head_layer_indices})

        elif self.vocab_projection_mode == VocabularyProjectionMode.SHARED_PROJECTION_CAST_OUTPUTS:
            # initialize layer-to-layer conversion matrices, that must be learned separately and aren't trainable params
            self.conv_matrix_name_suffix = f'_to_{self.num_layers}'
            self.name_to_conv_matrix = nn.ModuleDict(
                {f'{layer}{self.conv_matrix_name_suffix}': nn.Linear(self.output_size, self.output_size, bias=False)
                 for layer in config.lm_head_layer_indices if layer != self.num_layers})
            for conv_matrix_layer in self.name_to_conv_matrix.values():
                conv_matrix_layer.weight.requires_grad = False  # these matrices are trained externally

        # set inference-time generation parameters
        self.use_original_head = use_original_head
        self.output_layer_index = output_layer_index
        self.contrast_layer_indices = contrast_layer_indices
        self.contrast_function = contrast_function

        desc = 'original head' if self.use_original_head else (
            f'output layer {self.output_layer_index}' if not self.contrast_layer_indices
            else f'contrast between layers {self.contrast_layer_indices}')
        print(f'********* Using {desc} for generation (Mode={self.vocab_projection_mode.name}) ***********')

        if freeze_parameters:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)

    def _calculate_all_layer_outputs(self, original_exit_output: torch.FloatTensor,
                                     all_hidden_states: tuple[torch.FloatTensor]) -> dict[Union[int, str],
                                                                                          torch.FloatTensor]:
        device = original_exit_output.device

        # We assume that index 0 of *all_hidden_states* are the input embeddings, index 1 is the 1st layer,
        # index 2 is the 2nd, etc.
        index_to_layer_lm_logits = {}

        if self.vocab_projection_mode == VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION:
            for head_name, layer_lm_head in self.name_to_lm_exit_head.items():
                if head_name == 'lm_head_original':
                    index_to_layer_lm_logits['original'] = original_exit_output
                else:
                    layer_index = int(head_name.split(self.lm_head_name_prefix)[-1])
                    is_last_layer = layer_index == self.num_layers
                    layer_outputs = self.normalize_layer_output(all_hidden_states[layer_index], is_last_layer)
                    layer_outputs = layer_lm_head.to(device)(layer_outputs) if self.apply_lm_head \
                        else layer_outputs
                    index_to_layer_lm_logits[layer_index] = layer_outputs

        elif self.vocab_projection_mode == VocabularyProjectionMode.SHARED_PROJECTION_CAST_OUTPUTS:
            index_to_layer_lm_logits[self.num_layers] = original_exit_output
            for conv_name, conv_matrix in self.name_to_conv_matrix.items():
                layer_index = int(conv_name.split('_')[-3])
                layer_outputs = self.normalize_layer_output(all_hidden_states[layer_index], is_last_layer=False)
                if self.apply_lm_head:
                    # apply conversion matrix to the layer outputs to approximate final layer outputs
                    layer_outputs = conv_matrix.to(device)(layer_outputs)
                    layer_outputs = self.lm_head.to(device)(layer_outputs)
                index_to_layer_lm_logits[layer_index] = layer_outputs

        elif self.vocab_projection_mode == VocabularyProjectionMode.SHARED_PROJECTION_DIRECT:
            for layer_index in self.config.lm_head_layer_indices:
                is_last_layer = layer_index == self.num_layers
                layer_outputs = self.normalize_layer_output(all_hidden_states[layer_index], is_last_layer)
                if self.apply_lm_head:
                    layer_outputs = self.lm_head.to(device)(layer_outputs)
                index_to_layer_lm_logits[layer_index] = layer_outputs

        return index_to_layer_lm_logits

    def _calculate_model_output(self, original_exit_output: torch.FloatTensor,
                                index_to_layer_lm_logits: dict[Union[int, str], torch.FloatTensor]) \
            -> torch.FloatTensor:
        # according to the initialization, we decide which layers(s) are used for generating outputs at inference time
        if self.use_original_head:
            output_logits = original_exit_output
        elif not self.contrast_layer_indices:
            output_logits = index_to_layer_lm_logits[self.output_layer_index]
        else:
            lm_logits_upper = original_exit_output if self.contrast_layer_indices[0] == 'original' \
                else index_to_layer_lm_logits[self.contrast_layer_indices[0]]
            lm_logits_lower = index_to_layer_lm_logits[self.contrast_layer_indices[1]]
            output_logits = self.contrast_function(lm_logits_upper, lm_logits_lower)
        return output_logits

    def _forward(self, *args, **kwargs):
        """
        Apply multi-exit functionality on top of the original *forward()* method of the model
        """
        outputs = super().forward(*args, **kwargs)

        all_hidden_states = outputs.decoder_hidden_states if self.config.is_encoder_decoder else outputs.hidden_states
        index_to_layer_lm_logits = self._calculate_all_layer_outputs(original_exit_output=outputs.logits,
                                                                     all_hidden_states=all_hidden_states)

        outputs.logits = self._calculate_model_output(original_exit_output=outputs.logits,
                                                      index_to_layer_lm_logits=index_to_layer_lm_logits)
        return outputs

    @abstractmethod
    def normalize_layer_output(self, layer_output: torch.FloatTensor, is_last_layer: bool) -> torch.FloatTensor:
        """
        Model-specific method. Perform the calculations that are applied to the hidden states of the decoder
        before the projection to the vocabulary. Since many models already apply these steps *to the final
        layer outputs* within their decoder implementation, the behavior for the final layer and for intermediate
        hidden layers may differ.
        """

    @property
    @abstractmethod
    def output_size_config_key(self):
        """
        As different model configs use different names for the dimension of model output representations, each
        model class must specify the applicable name.
        """

    @property
    @abstractmethod
    def num_layers_config_key(self):
        """
        As different model configs use different names for the number of decoder layers, each model class
        must specify the applicable name.
        """
