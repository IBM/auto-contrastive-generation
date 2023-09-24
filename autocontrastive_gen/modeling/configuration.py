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
import ast
import dataclasses
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Union

from autocontrastive_gen.contrast_calculation import calculate_contrasted_logits


class VocabularyProjectionMode(Enum):
    LAYER_SPECIFIC_PROJECTION = 0
    SHARED_PROJECTION_CAST_OUTPUTS = 1
    SHARED_PROJECTION_DIRECT = 2


@dataclass
class MultiExitConfiguration:
    vocab_projection_mode: VocabularyProjectionMode = VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION
    lm_head_layer_indices: tuple[int, ...] = None
    # training
    freeze_parameters: bool = False
    # inference
    use_original_head: bool = True
    output_layer_index: int = None
    contrast_layer_indices: tuple[Union[int, str], int] = None
    contrast_function: Callable = partial(calculate_contrasted_logits, minimum_candidates=1, alpha=0.1)
    
    def __post_init__(self):
        if type(self.vocab_projection_mode) == str:
            self.vocab_projection_mode = VocabularyProjectionMode[self.vocab_projection_mode.upper()]
        if type(self.contrast_layer_indices) == str:
            self.contrast_layer_indices = tuple(int(idx) if idx != 'original' else 'original'
                                                for idx in self.contrast_layer_indices.split(';'))
        if type(self.lm_head_layer_indices) == str:
            self.lm_head_layer_indices = tuple(int(idx) for idx in self.lm_head_layer_indices.split(';'))
        if type(self.output_layer_index) == str:
            self.output_layer_index = int(self.output_layer_index)
        if type(self.use_original_head) == str:
            self.use_original_head = ast.literal_eval(self.use_original_head)
        
        # validate the generation parameters
        if self.contrast_layer_indices is not None:
            if self.use_original_head:
                raise Exception(f'Contradiction in model configuration: trying to use the original LM head '
                                f'but also to calculate contrast between layers {self.contrast_layer_indices}')

    def get_description(self):
        if self.use_original_head:
            return 'original_head'
        elif self.contrast_layer_indices:
            return f'{self.contrast_layer_indices[0]}_vs_{self.contrast_layer_indices[1]}'
        else:
            return f'layer_{self.output_layer_index}'
        
    def get_runtime_kwargs(self):
        """
        Returns the extra runtime arguments that are passed to the multi-exit model class, i.e., excluding the LM exit
        head indices, which are a property of the given model and are stored in the model config
        """
        kwargs = dataclasses.asdict(self)
        kwargs.pop('lm_head_layer_indices')
        return kwargs
