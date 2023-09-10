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
from argparse import ArgumentParser

from autocontrastive_gen.data_processing.dataset_catalog import DatasetsCatalog
from autocontrastive_gen.head_training.head_training_utils import get_head_training_function
from autocontrastive_gen.modeling.configuration import MultiExitConfiguration, VocabularyProjectionMode
from autocontrastive_gen.utils import get_model, get_tokenizer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=DatasetsCatalog.all_datasets())
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--lm_head_layer_indices', type=ast.literal_eval, default=(24, 22, 18, 12))
    parser.add_argument('--max-train-instance', type=int, default=None)
    args = parser.parse_args()

    print(args)
    
    dataset = getattr(DatasetsCatalog, args.dataset)
    dataset_dict = dataset.load()

    tokenizer = get_tokenizer(args.model_name_or_path, max_seq_length=args.max_seq_length)

    # for training, we can mostly use the default generation params of *MultiExitConfiguration* as we only care
    # about the loss, and not about which specific exit layer or decoding approach is used for generation outputs
    lm_config = MultiExitConfiguration(
        vocab_projection_mode=VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION,
        freeze_parameters=not dataset.is_seq2seq_task(),
        lm_head_layer_indices=args.lm_head_layer_indices,
    )

    model = get_model(args.model_name_or_path, lm_config)

    head_training_function = get_head_training_function(model_config=model.config,
                                                        is_seq2seq_task=dataset.is_seq2seq_task())

    print(f'***** Training LM heads using {head_training_function} *****')
    head_training_function(dataset_dict, model, tokenizer, args.max_seq_length,
                           output_dir=f'{args.output_dir}', debug=False, max_train_instance=args.max_train_instance)
