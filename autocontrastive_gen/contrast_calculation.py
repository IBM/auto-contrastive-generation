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


def expand_tensor(t, desired_shape):
    while len(t.shape) < len(desired_shape):
        t = t.unsqueeze(-1)

    t = t.expand(desired_shape)
    return t


def calculate_contrasted_logits(upper_layer_logits, lower_layer_logits, minimum_candidates=1, alpha=0.1):
    from autocontrastive_gen.utils import device

    lm_logits_upper = upper_layer_logits.softmax(dim=-1)
    lm_logits_lower = lower_layer_logits.softmax(dim=-1)

    # we set a probability threshold relative to the top candidate probability
    plausible_token_probability_threshold = \
        lm_logits_upper.max(-1).values.squeeze(-1) * torch.tensor(alpha)

    # when minimum_candidates=1, min_threshold will simply equal the plausible_token_probability_threshold
    min_threshold = torch.min(plausible_token_probability_threshold,
                              lm_logits_upper.sort(descending=True).values.squeeze()[..., minimum_candidates - 1])

    zero = torch.tensor(0.0).to(device)
    minus_inf = torch.tensor(-torch.inf).to(device)

    # for tokens above the threshold, calculate softmax of contrast score between lm_logits_upper and lm_logits_lower
    min_threshold_expanded = expand_tensor(min_threshold, lm_logits_upper.shape)
    contrasted_logits = torch.where(lm_logits_upper >= min_threshold_expanded,
                                    torch.log(lm_logits_upper) - torch.log(lm_logits_lower),
                                    lm_logits_upper)
    softmax_for_included_new = torch.where(lm_logits_upper >= min_threshold_expanded,
                                           contrasted_logits, minus_inf).softmax(-1)
    # calculate the total probability mass of tokens above the threshold
    sum_for_included_orig = torch.where(lm_logits_upper >= min_threshold_expanded,
                                        lm_logits_upper, zero).sum(-1)
    # redistribute this probability mass using the contrastive softmax scores
    sum_for_included_orig_expanded = expand_tensor(sum_for_included_orig, softmax_for_included_new.shape)
    adjusted_contrasted_logits = softmax_for_included_new * sum_for_included_orig_expanded
    contrasted_logits = torch.where(lm_logits_upper >= min_threshold_expanded,
                                    adjusted_contrasted_logits, lm_logits_upper)

    contrasted_logits = torch.log(contrasted_logits)
    return contrasted_logits
