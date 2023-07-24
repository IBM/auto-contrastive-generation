# [![version](https://img.shields.io/pypi/v/autocontrastive-gen)](https://pypi.org/project/autocontrastive-gen/)  ![license](https://img.shields.io/github/license/IBM/auto-contrastive-generation)  ![python](https://img.shields.io/badge/python-3.9%20|%203.10-blue)
# Auto-Contrastive Generation

Code to experiment with **multi-exit** text generation, and to reproduce the **Auto-Contrastive Decoding** experiments from [Gera et al. (2023)](#reference). 

Using this library you can:

1. Run inference on multi-exit generative language models, either using a *specific model exit layer*, or contrasting between model layers with *Auto-Contrastive Decoding*;
2. Train new multi-exit models, either for language modeling or for a specific task;
3. Try out new methods and algorithms for combining and contrasting the outputs of different model layers.


**Table of contents**

[Quick start](#quick-start)

[Setting the Multi-Exit Configuration](#setting-the-multi-exit-configuration)

[Running language modeling benchmarks](#running-language-modeling-benchmarks)

[Pre-trained model checkpoints](#pre-trained-model-checkpoints)

[Reference](#reference)

[License](#license)

## Quick start
1. Install the library with `pip install autocontrastive-gen`
2. Choose your desired [multi-exit parameters](#setting-the-multi-exit-configuration) - which model exit layer(s) do you want to use, and how?
3. Load a pre-trained multi-exit model and use it however you wish, within your own workflow

For instance, the following code will initialize the multi-exit [GPT2 model we release](#pre-trained-model-checkpoints) to use Auto-Contrastive Decoding:
```python
from autocontrastive_gen.modeling.configuration import MultiExitConfiguration
from autocontrastive_gen.modeling.auto_model import AutoMultiExitModel

# initialize a pre-trained multi-exit model to use auto-contrast between layer 24 and layer 12
multi_exit_config = MultiExitConfiguration(use_original_head=False, 
                                           contrast_layer_indices=(24, 12))
model = AutoMultiExitModel.from_pretrained("IBM/gpt2-medium-multiexit", multi_exit_config=multi_exit_config)
```

Then, the initialized model can be used as usual through the `transformers` library. For example, this code will run text generation using the initialized model:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("IBM/gpt2-medium-multiexit")
prompt = tokenizer("humpty dumpty sat on", return_tensors='pt')
generated_ids = model.generate(**prompt, max_new_tokens=15)
print(tokenizer.batch_decode(generated_ids))
```
Similarly, you can **train the model** with the `transformers` library just as you would do with any other model.

Note that the model behavior in inference and/or training depends on your [choice of parameters](#setting-the-multi-exit-configuration) when initializing the `MultiExitConfiguration`.

## Setting the Multi-Exit Configuration
Model behavior is determined by the `MultiExitConfiguration` used to initialize it. Most of the the configuration parameters are related to inference-time text generation behavior, but some are relevant for model training as well. The following parameters can be set:

- `lm_head_layer_indices: Tuple[int, ...]`: the indices of model layers which are connected to language modeling exit heads. As this is a basic characteristic of the model, this parameter only needs to be set *once* for initial pre-training of these exit heads. *Otherwise (e.g., when using one of the [released models](#pre-trained-model-checkpoints)), there is no need to specify this parameter as it is read directly from the model's config file*.
- `freeze_parameters: bool`: whether to freeze the language model parameters when training the model. You may wish to set this to True if training new exit layers for an existing (single-exit) pre-trained model checkpoint, but the default (`False`) is applicable for most use cases.
- `use_original_head: bool`: whether to use the original language modeling head of the pre-trained checkpoint for text generation. Setting this parameter to `True` *turns off all special model behavior and ignores the additional exit heads*, and thus also renders the parameters below irrelevant.
- `output_layer_index: int`: choose a *single* specific model exit for generation instead of the original (top layer) exit head.
- `contrast_layer_indices: Tuple[Union[int, str], int]`: use an *auto-contrastive generation* setting, contrasting between the two specified exit layers. To perform contrast with the original LM head, pass the string 'original'. For example: `(24, 18)` will perform contrast between the exits at layers 24 and 18, and `('original', 12)` will perform contrast between the original LM head and the head at exit 12.
- `contrast_function: Callable` *(Advanced)*: enables setting a custom contrast function, that gets logits of next-token predictions from two exit heads (i.e., those specified in `contrast_layer_indices`) and returns a modified set of predictions. By default, the `calculate_contrasted_logits` function is used, which performs the contrast calculation described in [Gera et al. (2023)](#reference).


## Running language modeling benchmarks
For GPT-family models, it is possible to run benchmarks for a given multi-exit model with the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) library, for different generation settings.

This is done via `python -m autocontrastive_gen.evaluation.lm_eval_harness.run_lm_eval`, by specifying `--model multi_exit_gpt` and adding the desired [multi-exit configuration settings](#setting-the-multi-exit-configuration) to the `--model_args` runtime argument.

_For example:_
```powershell
python -m autocontrastive_gen.evaluation.lm_eval_harness.run_lm_eval \
--tasks lambada_openai \
--model multi_exit_gpt \
--model_args pretrained=IBM/gpt2-medium-multiexit,use_original_head=False,contrast_layer_indices='original;12' \
--output_path my_output_path
```
_For details on the benchmark tasks available and on additional runtime arguments of the evaluation script, refer to https://github.com/EleutherAI/lm-evaluation-harness#basic-usage._


## Pre-trained model checkpoints
We release the following model checkpoints:
- [**ibm/gpt2-medium-multiexit**](https://huggingface.co/ibm/gpt2-medium-multiexit) - identical to the [GPT-2 Medium](https://huggingface.co/gpt2-medium) pre-trained model checkpoint, but with 12 newly-trained linear exit heads.
- [**ibm/gpt-neo-125m-multiexit**](https://huggingface.co/ibm/gpt-neo-125m-multiexit) - identical to the [GPT Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125m) pre-trained model checkpoint, but with 6 newly-trained linear exit heads.

The new heads were trained on the English portion of the [CC-100](https://huggingface.co/datasets/cc100) dataset. For further details, refer to Appendix A of the paper. 

## Reference
Ariel Gera, Roni Friedman, Ofir Arviv, Chulaka Gunasekara, Benjamin Sznajder, Noam Slonim and Eyal Shnarch (2023). 
[The Benefits of Bad Advice: Autocontrastive Decoding across Model Layers](https://aclanthology.org/2023.acl-long.580). ACL 2023.

Please cite: 
```
@inproceedings{gera2023autocontrastive,
  title={The Benefits of Bad Advice: Autocontrastive Decoding across Model Layers},
  author={Gera, Ariel and Friedman, Roni and Arviv, Ofir and Gunasekara, Chulaka and Sznajder, Benjamin and Slonim, Noam and Shnarch, Eyal},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month={july},
  address={Toronto, Canada},
  year={2023},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/2023.acl-long.580},
  pages={10406--10420}
}
```

## License
This work is released under the Apache 2.0 license. The full text of the license can be found in [LICENSE](LICENSE).
