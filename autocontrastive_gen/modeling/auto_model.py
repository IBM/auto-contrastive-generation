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
from transformers import AutoConfig, GPTNeoConfig, GPT2Config, T5Config

from autocontrastive_gen.modeling.configuration import MultiExitConfiguration
from autocontrastive_gen.modeling.gpt2_multi_head import MultiHeadGPT2
from autocontrastive_gen.modeling.gpt_neo_multi_head import MultiHeadGPTNeo
from autocontrastive_gen.modeling.t5_multi_head import MultiHeadT5


class AutoMultiExitModel:
    @staticmethod
    def from_pretrained(model_name_or_path, multi_exit_config: MultiExitConfiguration, **extra_kwargs):
        # Determine the appropriate multi-head model class according to the standard model config it is based on
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        if type(model_config) == GPTNeoConfig:
            model_class = MultiHeadGPTNeo
        elif type(model_config) == GPT2Config:
            model_class = MultiHeadGPT2
        elif type(model_config) == T5Config:
            model_class = MultiHeadT5
        else:
            raise Exception(f'Model {model_name_or_path} of type {type(model_config)} is not supported')

        model_config.output_hidden_states = True
        # If loading a standard single-exit model checkpoint, add the desired exit layers to the model config
        # (this parameter should be set once, at training time; at inference it is loaded from the model config file)
        if not hasattr(model_config, 'lm_head_layer_indices'):
            model_config.lm_head_layer_indices = multi_exit_config.lm_head_layer_indices

        elif multi_exit_config.use_original_head is False:
            # validate multi-exit config is compatible with the model checkpoint
            multi_exit_config_layers = multi_exit_config.contrast_layer_indices \
                if multi_exit_config.contrast_layer_indices is not None else [multi_exit_config.output_layer_index]
            for layer in multi_exit_config_layers:
                if layer not in {*model_config.lm_head_layer_indices, 'original'}:
                    raise Exception(f'Exit layer {layer} in the MultiExitConfiguration does not match the exit heads '
                                    f'in the pre-trained model checkpoint ({model_config.lm_head_layer_indices})')

        multi_exit_kwargs = multi_exit_config.get_runtime_kwargs()
        return model_class.from_pretrained(model_name_or_path, config=model_config, **multi_exit_kwargs, **extra_kwargs)
