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
from typing import Mapping

from simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE


simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


def calc_metrics(prompt_text, generated_text) -> Mapping:
    try:
        rep_2, rep_3, rep_4, diversity = measure_repetition_and_diversity([generated_text])
    except ZeroDivisionError:  # text is too short
        diversity = 0

    coherence = simcse_model.similarity(generated_text, prompt_text)
    return {'diversity': diversity,
            'coherence': coherence}
