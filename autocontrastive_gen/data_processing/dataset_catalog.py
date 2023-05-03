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
from dataclasses import dataclass, field
from typing import Callable, Mapping

from datasets import load_dataset, DatasetDict

from autocontrastive_gen.data_processing.preprocessing_functions import wikitext_dataset_preprocess, \
    wikinews_dataset_preprocess, bookcorpus_dataset_preprocess


@dataclass
class Dataset:
    hf_dataset_name: str
    source_column: str
    target_column: str = None
    hf_subset_name: str = None
    dataset_preprocessing_func: Callable = None
    extra_kwargs: Mapping = field(default_factory=dict)

    def load(self) -> DatasetDict:
        dataset_dict = load_dataset(self.hf_dataset_name, self.hf_subset_name, **self.extra_kwargs)

        if self.dataset_preprocessing_func is not None:
            dataset_dict = dataset_dict.map(self.dataset_preprocessing_func,
                                            batched=True, remove_columns=list(dataset_dict['train'].column_names))

        dataset_dict = dataset_dict.rename_column(self.source_column, 'source_text')
        if self.target_column:
            dataset_dict = dataset_dict.rename_column(self.target_column, 'target_text')

        if 'validation' not in dataset_dict:
            dataset_dict['validation'] = dataset_dict['train']

        return dataset_dict

    def is_seq2seq_task(self):
        return self.target_column is not None


"""
To create an auth token generate a token in https://huggingface.co/settings/tokens; This is required for loading the
WikiNews dataset.
"""
HF_AUTH_TOKEN = None


class DatasetsCatalog:
    wikitext_103 = Dataset('wikitext', hf_subset_name='wikitext-103-raw-v1', source_column='text',
                           dataset_preprocessing_func=wikitext_dataset_preprocess)
    wikinews = Dataset('bigscience-data/roots_en_wikinews', source_column='text',
                       dataset_preprocessing_func=wikinews_dataset_preprocess,
                       extra_kwargs={'use_auth_token': HF_AUTH_TOKEN})  # Note: requires setting HF_AUTH_TOKEN
    bookcorpus = Dataset('bookcorpus', source_column='text', dataset_preprocessing_func=bookcorpus_dataset_preprocess)
    cc_en = Dataset('cc100', source_column='text', extra_kwargs={'lang': 'en'})

    @staticmethod
    def all_datasets():
        return [var for var in vars(DatasetsCatalog)
                if not var.startswith('__') and not callable(getattr(DatasetsCatalog, var))]
