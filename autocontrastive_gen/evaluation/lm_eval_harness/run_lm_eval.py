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
from lm_eval.__main__ import cli_evaluate


if __name__ == '__main__':
    from autocontrastive_gen.evaluation.lm_eval_harness.lm_eval_multi_exit import HFLMMultiExit
    
    
    cli_evaluate()