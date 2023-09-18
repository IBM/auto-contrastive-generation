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
import random

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig, T5Config

from autocontrastive_gen.data_processing.dataset_catalog import DatasetsCatalog
from autocontrastive_gen.modeling.auto_model import AutoMultiExitModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(model_name_or_path, multi_exit_config, vanilla_model: bool = False):
    if vanilla_model:
        if type(AutoConfig.from_pretrained(model_name_or_path)) == T5Config:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    else:
        model = AutoMultiExitModel.from_pretrained(model_name_or_path, multi_exit_config=multi_exit_config).to(device)

    return model


def get_tokenizer(model_name, max_seq_length=512):
    tokenizer_params = {'pad_token': '<|endoftext|>'} if 'gpt' in model_name else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_length, **tokenizer_params)
    return tokenizer


def get_texts_for_inference(dataset, num_samples, dataset_split):
    dataset = getattr(DatasetsCatalog, dataset)
    dataset_dict = dataset.load()

    texts = dataset_dict[dataset_split]['source_text']

    is_seq2seq_task = dataset.is_seq2seq_task()
    if is_seq2seq_task:
        targets = dataset_dict[dataset_split]['target_text']
    else:
        targets = ['dummy_target'] * len(texts)

    if num_samples is not None and num_samples < len(texts):
        sample_ids = random.Random(0).sample(list(range(len(texts))), k=num_samples)
        texts = [texts[i] for i in sample_ids]
        targets = [targets[i] for i in sample_ids]

    return is_seq2seq_task, texts, targets
