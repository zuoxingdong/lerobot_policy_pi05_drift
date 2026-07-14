#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drift changes nothing about pre/post-processing, so this module delegates to LeRobot's
in-tree Pi0.5 processor factory. Reusing it (rather than vendoring a copy) keeps the tokenizer
step registered under the stock ``pi05_prepare_state_tokenizer_processor_step`` name, so
processor pipelines saved by this plugin — and checkpoints trained with in-tree Pi0.5 — load
interchangeably with no conversion.

The module name and function name are load-bearing: LeRobot's plugin factory resolves
``make_{config.type}_pre_post_processors`` from the module found by rewriting the config
module's ``configuration_`` prefix to ``processor_``.
"""

from typing import Any

import torch

# Importing the in-tree module also registers Pi05PrepareStateTokenizerProcessorStep
# under its stock registry name.
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_pi05_drift import PI05DriftConfig


def make_pi05_drift_pre_post_processors(
    config: PI05DriftConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_pi05_pre_post_processors(config, dataset_stats)
