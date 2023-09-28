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
from transformers import LlamaForCausalLM

from autocontrastive_gen.modeling.multi_exit_modeling import MultiExitModel


class MultiExitLlama(MultiExitModel, LlamaForCausalLM):
    output_size_config_key = 'hidden_size'
    num_layers_config_key = 'num_hidden_layers'

    def normalize_layer_output(self, layer_output, is_last_layer: bool):
        if not is_last_layer:  # layer norm for top layer is already applied within the Llama code
            layer_output = self.model.norm(layer_output)
        return layer_output
