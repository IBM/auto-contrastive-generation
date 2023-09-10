from typing import Optional, Union, Callable

import torch
from torch import nn

from autocontrastive_gen.modeling.configuration import VocabularyProjectionMode


class MultiExitMixin:
    """
    This Mixin can be used to add multi-exit functionality to a model with a language modeling head, 
    i.e., add the capability to produce next-token predictions from one of the intermediate model layers
    or by contrasting the outputs of different layers.
    The MultiExitMixin provides methods for initialization and inference that are common across models. These
    should be called from within the __init__() and forward() methods of each multi-exit model class.
    """
    vocab_projection_mode: VocabularyProjectionMode
    output_size: int
    num_layers: int
    apply_lm_head: bool

    name_to_lm_exit_head: Optional[nn.ModuleDict]  # for VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION
    name_to_conv_matrix: Optional[nn.ModuleDict]  # for VocabularyProjectionMode.SHARED_PROJECTION_CAST_OUTPUTS
    
    lm_head_name_prefix = 'lm_head_'
    conv_matrix_name_suffix = Optional[str]
    
    use_original_head: bool
    output_layer_index: int
    contrast_layer_indices: Optional[tuple[Union[int, str], int]]
    contrast_function: Optional[Callable]

    def initialize_multi_exit_model(self, config, output_size, num_layers, vocab_projection_mode: VocabularyProjectionMode,
                                    use_original_head=False, output_layer_index=24, contrast_layer_indices=None, 
                                    contrast_function=None, freeze_parameters=True):
        if not hasattr(config, 'lm_head_layer_indices'):
            raise Exception(f"LM exit head indices must be specified when initializing {self.__class__.__name__}")
        
        self.vocab_projection_mode = vocab_projection_mode
        self.output_size = output_size
        self.num_layers = num_layers
        
        # the LM head can be turned off for the purposes of learning layer-to-layer conversion matrices
        self.apply_lm_head = True

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
    
    def calculate_all_layer_outputs(self, original_exit_output: torch.FloatTensor, 
                                    all_hidden_states: tuple[torch.FloatTensor], device: torch.device) \
            -> dict[Union[int, str], torch.FloatTensor]:
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
    
    def calculate_model_output(self, original_exit_output: torch.FloatTensor, 
                               index_to_layer_lm_logits: dict[Union[int, str], torch.FloatTensor]) -> torch.FloatTensor:
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
                
    def normalize_layer_output(self, layer_output: torch.FloatTensor, is_last_layer: bool) -> torch.FloatTensor:
        """
        Model-specific method. Perform the normalizations that are used in the given model 
        before passing outputs to the language modeling head. Since many models already 
        apply these normalizations to the final layer outputs within their decoder implementation, 
        the behavior for the final model layer and for intermediate hidden layers may differ.
        """
