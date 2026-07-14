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

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot.policies.rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_drift")
@dataclass
class PI05DriftConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # Drifting (one-step generative) objective, shared with the SmolVLA-Drift
    # plugin -- see `drifting_util.py`. When enabled, training uses the
    # drift loss instead of flow matching, and `sample_actions` dispatches to a
    # one-step direct-read sampler (`num_inference_steps` is never read).
    use_drifting_loss: bool = True
    drifting_gen_per_label: int = 8
    drifting_temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_per_timestep_loss: bool = False
    drifting_perdim_loss: bool = True

    # Initialization: load pretrained PaliGemma/VLM weights from
    # `pretrained_path` but skip + freshly reinitialize the action expert,
    # action projections, and time MLPs. Consumed (and reset to False) by
    # `PI05DriftPolicy.from_pretrained` so trained checkpoints reload cleanly;
    # `init_label` keeps the provenance in the saved config.
    fresh_action_expert: bool = False
    init_label: str | None = None

    # KeyStone test-time self-consistency selection (drift path only). With
    # `test_time_samples` K > 1, inference draws K one-step candidate chunks and
    # returns the guarded cluster-medoid (see `keystone_util.py`). K=1 = off.
    test_time_samples: int = 1
    test_time_clusters: int = 2
    test_time_unimodal_tau: float = 0.3

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.use_drifting_loss:
            if self.drifting_gen_per_label < 2:
                raise ValueError(
                    "`drifting_gen_per_label` (G) must be >= 2; G=1 degenerates the drift objective "
                    "(no sibling samples to repel from)."
                )
            try:
                drifting_temperatures = tuple(float(t) for t in self.drifting_temperatures)
            except TypeError as exc:
                raise ValueError("`drifting_temperatures` must be a non-empty iterable of floats.") from exc
            if len(drifting_temperatures) == 0:
                raise ValueError("`drifting_temperatures` must contain at least one positive value.")
            for temperature in drifting_temperatures:
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError(
                        "`drifting_temperatures` must contain only finite positive values; "
                        f"got {self.drifting_temperatures}."
                    )
            self.drifting_temperatures = drifting_temperatures
            if self.drifting_perdim_loss and self.drifting_per_timestep_loss:
                raise ValueError(
                    "`drifting_perdim_loss=True` is mutually exclusive with "
                    "`drifting_per_timestep_loss=True`; set per-timestep loss to False."
                )
            if self.rtc_config is not None:
                raise ValueError(
                    "`rtc_config` is incompatible with `use_drifting_loss=True`: Real-Time Chunking "
                    "hooks into the multi-step flow-matching integrator, and the Drift sampler is a "
                    "single forward pass with no integrator to hook into."
                )
            # NOTE: `num_inference_steps` is a flow-matching integrator setting and
            # is never read on the drift path -- `sample_actions` dispatches to the
            # one-step drift sampler before the Euler loop. Deliberately untouched.

        # KeyStone test-time selection validation.
        if self.test_time_samples < 1:
            raise ValueError(f"`test_time_samples` must be >= 1, got {self.test_time_samples}.")
        if self.test_time_samples > 1:
            if not self.use_drifting_loss:
                raise ValueError(
                    "`test_time_samples` > 1 (KeyStone) requires `use_drifting_loss=True`: candidate "
                    "selection hooks into the one-step drift sampler, not the flow-matching integrator."
                )
            if self.rtc_config is not None:
                raise ValueError(
                    "`test_time_samples` > 1 is incompatible with `rtc_config`: RTC guides the "
                    "denoising trajectory, KeyStone selects among independent one-step candidates."
                )
            if self.test_time_clusters < 2:
                raise ValueError(
                    f"`test_time_clusters` must be >= 2 when selection is on, got {self.test_time_clusters}."
                )
            if not self.test_time_unimodal_tau > 0:
                raise ValueError(
                    f"`test_time_unimodal_tau` must be > 0, got {self.test_time_unimodal_tau}."
                )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
