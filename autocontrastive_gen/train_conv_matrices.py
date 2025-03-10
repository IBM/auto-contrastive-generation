#
#  Copyright (c) 2025 IBM Corp.
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
import pickle
import random
from argparse import ArgumentParser

import torch
from sklearn.linear_model import LinearRegression as LinReg
from transformers import AutoConfig
from tqdm.auto import tqdm

from autocontrastive_gen.data_processing.dataset_catalog import DatasetsCatalog
from autocontrastive_gen.modeling.configuration import MultiExitConfiguration, VocabularyProjectionMode
from autocontrastive_gen.utils import device, get_tokenizer, get_model, get_texts_for_inference


"""
The code below trains conversion matrices between the outputs of different model layers, as proposed
by Din et al. 2023 (https://arxiv.org/abs/2303.09435, https://github.com/sashayd/mat)
"""


def linreg(x, y, intercept=False, file_name=None):
    orig_device = x.device

    x = x.detach().cpu()
    y = y.detach().cpu()

    reg = LinReg(fit_intercept=intercept).fit(x, y)
    if intercept:
        reg = [torch.from_numpy(reg.coef_),
               torch.from_numpy(reg.intercept_)]
    else:
        reg = torch.from_numpy(reg.coef_)

    if file_name is not None:
        with open(file_name, 'wb') as f:
            pickle.dump(reg, f)

    if intercept:
        reg = [c.to(orig_device) for c in reg]
    else:
        reg = reg.to(orig_device)

    return reg


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, required=True)
    parser.add_argument('-e', '--exit_indices_to_train', nargs='+', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='wikitext_103', choices=DatasetsCatalog.all_datasets())
    parser.add_argument('--num_examples', type=int, default=5000)
    parser.add_argument('--num_vectors_per_example', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=512)
    args = parser.parse_args()

    _, texts, _ = get_texts_for_inference(args.dataset, args.num_examples, 'train')

    tokenizer = get_tokenizer(args.model_name_or_path, max_seq_length=args.max_seq_length)
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    last_layer = model_config.num_decoder_layers if model_config.is_encoder_decoder else model_config.num_hidden_layers
    if args.exit_indices_to_train is not None:
        exit_indices_to_train = args.exit_indices_to_train
    else:
        # train matrices for all the layers
        exit_indices_to_train = range(1, last_layer)

    layers_to_infer = [last_layer, *exit_indices_to_train]

    if hasattr(model_config, 'lm_head_layer_indices'):
        model_exit_indices = {*layers_to_infer, *model_config.lm_head_layer_indices}
    else:
        model_exit_indices = layers_to_infer

    lm_config = MultiExitConfiguration(
        vocab_projection_mode=VocabularyProjectionMode.SHARED_PROJECTION_CAST_OUTPUTS,
        lm_head_layer_indices=tuple(sorted(model_exit_indices)),
        use_original_head=False,
        output_layer_index=last_layer,
    )

    model = get_model(args.model_name_or_path, lm_config)

    # instead of outputting logits, we want the layer outputs *before* the LM head
    setattr(model, 'apply_lm_head', False)

    layer_to_vectors = {}
    top_layer_sequences = {}
    for i, layer in enumerate(layers_to_infer):
        setattr(model, 'output_layer_index', layer)

        layer_output_vectors = []
        print(f'generating texts from {len(texts)} prompts')
        for idx, text in tqdm(enumerate(texts), total=len(texts)):
            prompt = tokenizer(text, return_tensors='pt', truncation=True).input_ids.to(device)
            num_tokens_to_keep = random.Random(idx).randint(1, prompt.shape[1])
            prompt = prompt[:, :num_tokens_to_keep]

            if layer == last_layer:
                setattr(model, 'apply_lm_head', True)
                # generate a few tokens, for a kind of "teacher forcing" of final layer decoder generation behavior
                generated = model.generate(prompt,
                                           max_new_tokens=args.num_vectors_per_example,
                                           num_beams=1, do_sample=False, top_p=1.0, top_k=0.0,
                                           return_dict_in_generate=True,
                                           output_hidden_states=True)
                top_layer_sequences[idx] = generated.sequences[:, :-1]

                # extract the output vectors
                generated_hidden_states = generated.decoder_hidden_states if model_config.is_encoder_decoder \
                    else generated.hidden_states
                for hidden_states in generated_hidden_states:
                    v = hidden_states[-1][:, -1, :]  # output of the final decoder layer given the preceding sequence
                    layer_output_vectors.append(model.normalize_layer_output(v, True).detach().cpu().squeeze())

            else:
                setattr(model, 'apply_lm_head', False)
                if model_config.is_encoder_decoder:
                    model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(prompt, {}, 'input_ids')
                    outputs = model(decoder_input_ids=top_layer_sequences[idx], **model_kwargs)
                    layer_output_vectors.extend(list(outputs.logits.detach().cpu().squeeze()))
                else:
                    for j in range(args.num_vectors_per_example):
                        model_output = model(top_layer_sequences[idx][:, :prompt.shape[1]+j])
                        layer_output_vectors.append(model_output.logits[:, -1, :].detach().cpu().squeeze())

        layer_to_vectors[layer] = torch.vstack(layer_output_vectors)

        if layer == last_layer:
            continue

        print(f'learning conversion matrix from layer {layer} to layer {last_layer}, based on {layer_to_vectors[layer].shape[0]} vectors')
        mat = linreg(layer_to_vectors[layer], layer_to_vectors[last_layer], intercept=False,
                     file_name=f'{layer}_{last_layer}.pickle')

        # save model with the conversion matrices
        model.name_to_conv_matrix[f'{layer}_to_{last_layer}'].weight.data[:] = mat

        out_path = args.model_name_or_path.replace('/', '_') + '_multi_exit_' + '-'.join(str(l) for l in layers_to_infer[:i+1])
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
