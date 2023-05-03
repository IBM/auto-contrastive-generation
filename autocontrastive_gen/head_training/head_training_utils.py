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
from transformers import Trainer, TrainingArguments


def pretrain_tokenize_function(examples, tokenizer):
    # tokenization for pretraining-like fine-tuning, where the texts are chunked and we do not need
    # padding/truncation/attention mask
    return tokenizer([t for t in examples['source_text']], return_attention_mask=False, return_tensors='np')


def group_texts(examples, inputs_length):
    """
    Based on https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
    """
    # Concatenate all texts.
    concatenated_examples = {'input_ids': sum(examples['input_ids'], [])}
    total_length = len(concatenated_examples['input_ids'])
    # We drop the small remainder
    if total_length >= inputs_length:
        total_length = (total_length // inputs_length) * inputs_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + inputs_length] for i in range(0, total_length, inputs_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def train(model, tokenizer, train_dataset, output_dir, optimizer_name, learning_rate, lr_scheduler_type, data_collator,
          debug=False):

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=8,
        evaluation_strategy='steps', eval_steps=1000, save_strategy='epoch',
        optim=optimizer_name,
        lr_scheduler_type=lr_scheduler_type,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset.select(range(50)) if not debug else train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


def get_head_training_function(model_config, is_seq2seq_task):
    from autocontrastive_gen.head_training.gpt2_training import run_gpt2_pretraining, run_gpt2_downstream_training
    from autocontrastive_gen.head_training.t5_training import run_t5_pretraining, run_t5_seq2seq

    if model_config.is_encoder_decoder:  # T5
        if is_seq2seq_task:
            return run_t5_seq2seq
        else:
            return run_t5_pretraining
    else:
        if is_seq2seq_task:
            return run_gpt2_downstream_training
        else:
            return run_gpt2_pretraining
