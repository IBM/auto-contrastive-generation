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
from transformers import T5ForConditionalGeneration

from autocontrastive_gen.modeling.multi_exit_modeling import MultiExitModel


class MultiExitT5(MultiExitModel, T5ForConditionalGeneration):
    output_size_config_key = 'd_model'
    num_layers_config_key = 'num_decoder_layers'

    def normalize_layer_output(self, layer_output, is_last_layer):
        if not is_last_layer:  # layer norm and dropout for top layer are already applied within the T5 decoder
            layer_output = self.decoder.final_layer_norm(layer_output)
            layer_output = self.decoder.dropout(layer_output)
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            layer_output = layer_output * (self.model_dim ** -0.5)
        return layer_output
