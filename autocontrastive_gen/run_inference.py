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
import json
import os

from argparse import ArgumentParser

import pandas as pd
from tqdm.auto import tqdm

from autocontrastive_gen.data_processing.dataset_catalog import DatasetsCatalog
from autocontrastive_gen.evaluation.auto_metrics import calc_metrics
from autocontrastive_gen.modeling.configuration import MultiExitConfiguration, VocabularyProjectionMode
from autocontrastive_gen.utils import get_model, get_tokenizer, device, get_texts_for_inference


def run(args):
    is_seq2seq_task, texts, targets = get_texts_for_inference(args.dataset, args.number_of_samples, args.dataset_split)

    lm_config = MultiExitConfiguration(
        vocab_projection_mode=VocabularyProjectionMode.LAYER_SPECIFIC_PROJECTION,
        use_original_head=args.use_original_head,
        output_layer_index=args.output_layer_index,
        contrast_layer_indices=args.contrast_layer_indices,
    )

    tokenizer = get_tokenizer(args.model_name_or_path, max_seq_length=args.max_seq_length)
    model = get_model(args.model_name_or_path, lm_config, args.vanilla_model)
    modus = 'top_k' if args.use_top_k else ('top_p' if args.use_top_p else f'beam_{args.beam_search}')
    desc = f'{args.model_name_or_path.replace("/", "_")}_{lm_config.get_description()}_{modus}_{args.additional_desc}'

    all_results = []
    print(f'generating texts from {len(texts)} prompts')
    for i, (text, target) in tqdm(enumerate(zip(texts, targets)), total=len(texts)):
        if not is_seq2seq_task:
            prompt_text = ' '.join(text.split()[:args.num_prompt_tokens])
        else:
            prompt_text = text
        prompt = tokenizer(prompt_text, return_tensors='pt', truncation=True).input_ids.to(device)
        generated_ids = model.generate(prompt,
                                       pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       max_new_tokens=args.max_new_tokens,
                                       num_beams=args.beam_search,
                                       do_sample=args.use_top_k or args.use_top_p,
                                       top_p=0.95 if args.use_top_p else 1.0,
                                       top_k=50 if args.use_top_k else 0.0,
                                       output_hidden_states=True)

        if not model.config.is_encoder_decoder:  # keep only the newly-generated tokens
            generated_ids = generated_ids[:, prompt.shape[1]:]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        example_metrics = calc_metrics(prompt_text, generated_text)
        all_results.append({'text_id': i, 'model_description': desc, 'prompt': prompt_text,
                            'generated_text': generated_text, **example_metrics})

    if args.output_dir:
        output_dir = os.path.join(args.output_dir, desc)
    else:
        output_dir = os.path.join('output', desc)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, f'{desc}_{args.dataset}_output.csv'))
    print(f'csv with generation results written to {os.path.abspath(output_dir)}')

    metrics = {'diversity': df['diversity'].mean(), 'coherence': df['coherence'].mean(),
               'num_examples': len(df)}
    print(metrics)
    with open(os.path.join(output_dir, f'metrics_{args.dataset}{args.additional_desc}.json'), 'w') as f:
        f.write(json.dumps(metrics))

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=DatasetsCatalog.all_datasets())
    parser.add_argument('--max_seq_length', type=int, default=512)

    parser.add_argument('--use_original_head', type=ast.literal_eval, required=True)
    parser.add_argument('--output_layer_index', type=int, default=24)
    parser.add_argument('--contrast_layer_indices', type=str, default=None)
    parser.add_argument('--vanilla-model', action='store_true', default=False)
    
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_prompt_tokens', type=int, default=32)                                               
    parser.add_argument('--beam_search', type=int, default=1)
    parser.add_argument('--use_top_k', type=ast.literal_eval, default=False)
    parser.add_argument('--use_top_p', type=ast.literal_eval, default=False)

    parser.add_argument('--dataset_split', type=str, default='validation')
    parser.add_argument('--number_of_samples', type=int, default=None)

    parser.add_argument('--additional_desc', type=str, default='')
    parser.add_argument('--output_dir', '-o', type=str, required=False)
    args = parser.parse_args()
    run(args)
