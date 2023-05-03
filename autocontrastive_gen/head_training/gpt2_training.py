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
from typing import Optional

from datasets import DatasetDict

from autocontrastive_gen.head_training.head_training_utils import group_texts, pretrain_tokenize_function, train


def gpt_preprocess_func(examples, ignore_input_in_loss, pad_token_id=50256):
    # for autoregressive generation, the concatenated source and target also serve as the label, and we can choose
    # whether to include the input tokens in the loss calculation
    if ignore_input_in_loss:
        # in the labels we replace the source input ids and the padding with -100 so that they won't be included in the loss
        labels = []
        for concatenated, target in zip(examples['input_ids'], examples['target_input_ids']):
            target_end_idx = concatenated.index(pad_token_id) if concatenated[-1] == pad_token_id else len(concatenated)
            target_start_idx = target_end_idx - len(target)
            labels.append([-100]*target_start_idx + concatenated[target_start_idx:target_end_idx] + [-100]*(len(concatenated)-target_end_idx))
        examples['labels'] = labels
    else:
        # in the labels we replace the padding ids with -100 so that they won't be included in the loss
        labels = [[input_id if input_id != pad_token_id else -100 for input_id in example_input_ids] 
                  for example_input_ids in examples['input_ids']]
        examples['labels'] = labels
    return examples


def run_gpt2_pretraining(dataset_dict: DatasetDict, model, tokenizer, max_seq_length, output_dir, debug=False,
                         max_train_instance: Optional[int] = None):
    column_names = dataset_dict["train"].column_names
    if debug:
        dataset_dict['train'] = dataset_dict['train'].select(range(1000))
    if max_train_instance:
        dataset_dict['train'] = dataset_dict['train'].select(range(max_train_instance))

    dataset_dict = dataset_dict.map(lambda examples: pretrain_tokenize_function(examples, tokenizer),
                                    batched=True, remove_columns=column_names)
    print(f"train dataset size before chunking: {len(dataset_dict['train'])}")
    # chunk different examples together up to max_seq_length
    dataset_dict = dataset_dict.map(lambda examples: group_texts(examples, max_seq_length), batched=True)

    dataset_dict = dataset_dict.map(lambda examples: gpt_preprocess_func(examples, ignore_input_in_loss=False),
                                    batched=True)
    print(f"final train dataset size: {len(dataset_dict['train'])}")

    train(model, tokenizer, train_dataset=dataset_dict['train'], output_dir=output_dir,
          optimizer_name='adamw_hf', learning_rate=2e-4, lr_scheduler_type='linear', data_collator=None,
          debug=debug)


def gpt_finetune_tokenize_function(examples, tokenizer, max_seq_length, ignore_input_in_loss):
    # tokenization for downstream task fine-tuning with a source and a target (label);
    # the source and target are concatenated to a single input sequence

    tokenized_dict = \
        tokenizer.batch_encode_plus(list(zip(examples['source_text'], examples['target_text'])),
                                    max_length=max_seq_length, padding='max_length', truncation='only_first',
                                    return_attention_mask=True, add_special_tokens=False)

    if ignore_input_in_loss:
        # we also tokenize the targets separately so we will know how to mask the sources in gpt_preprocess_func()
        target_tokenized = tokenizer.batch_encode_plus(examples['target_text'],
                                                       truncation=False,
                                                       return_attention_mask=False, add_special_tokens=False)

        tokenized_dict['target_input_ids'] = target_tokenized['input_ids']

    return tokenized_dict


def run_gpt2_downstream_training(dataset_dict: DatasetDict, model, tokenizer, max_seq_length, output_dir, debug=False,
                                 max_train_instance: Optional[int] = None):
    column_names = dataset_dict["train"].column_names
    
    if max_train_instance is not None:
        dataset_dict['train'] = dataset_dict['train'].select(range(max_train_instance))

    dataset_dict = dataset_dict.map(lambda examples:
                                    gpt_finetune_tokenize_function(examples, tokenizer, max_seq_length,
                                                                   ignore_input_in_loss=True),
                                    batched=True, remove_columns=column_names)
    dataset_dict = dataset_dict.map(lambda examples: gpt_preprocess_func(examples, ignore_input_in_loss=True),
                                    batched=True)

    train(model, tokenizer, train_dataset=dataset_dict['train'], output_dir=output_dir,
          optimizer_name='adamw_hf', learning_rate=5e-4, lr_scheduler_type='linear', data_collator=None)
