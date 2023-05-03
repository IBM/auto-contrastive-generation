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
import torch
import transformers

from lm_eval.models import gpt2

from autocontrastive_gen.modeling.auto_model import AutoMultiExitModel
from autocontrastive_gen.modeling.configuration import MultiExitConfiguration


class HFLMMultiExit(gpt2.HFLM):
    """
    Overrides the init() method from lm_eval.models.gpt2.HFLM (v0.3.0), with slight modifications so that the model
    will be initialized with a multi-exit model class rather than using the transformers AutoModelForCausalLM
    """
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2-medium",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        **multi_args
    ):
        super(gpt2.HFLM, self).__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        revision = revision + ("/" + subfolder if subfolder is not None else "")

        multi_exit_config = MultiExitConfiguration(**multi_args)

        self.gpt2 = AutoMultiExitModel.from_pretrained(
            pretrained,
            multi_exit_config=multi_exit_config,
            revision=revision,
        ).to(self.device)
        self.gpt2.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode("hello\n\nhello") == [
                31373,
                198,
                198,
                31373,
            ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size
