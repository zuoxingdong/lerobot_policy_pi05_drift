# lerobot_policy_pi05_drift

[![CI](https://github.com/zuoxingdong/lerobot_policy_pi05_drift/actions/workflows/ci.yml/badge.svg)](https://github.com/zuoxingdong/lerobot_policy_pi05_drift/actions/workflows/ci.yml)

A [LeRobot](https://github.com/huggingface/lerobot) plugin policy for **Pi0.5-Drift**.

Pi0.5 decodes an action chunk by integrating a flow-matching ODE, which takes 10 forward passes
of the action expert per chunk. Pi0.5-Drift trains the same network with a one-step **Drifting**
(DBPO) objective instead: a single forward pass maps noise directly to the chunk.

- **96.3%** average success over the four LIBERO suites, vs 93.1% for 10-step flow matching
  (**96.5%** with KeyStone)
- **1 NFE** per chunk — **85.6 ms** vs 259.8 ms per chunk at batch 1 (**~3× faster**)
- **Backbone-robust**: with the VLM frozen (only the 300M expert trained), Drift holds
  **96.3%** on LIBERO-Spatial while flow matching drops to 85.7%
- Same PaliGemma VLM + action expert as Pi0.5, byte-identical weight layout
- Optional **KeyStone** test-time selection (K one-step candidates, ~zero added latency)

Project website: <https://zuoxingdong.github.io/drift-vla/>

## Results

LIBERO success rates, 50 episodes/task × 3 eval seeds, closed-loop replanning (numbers from
the [project page](https://zuoxingdong.github.io/drift-vla/)):

| Policy | NFE | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|---:|
| **Pi0.5-Drift** | 1 | **98.3 ±0.3** | **98.1 ±0.9** | **95.6 ±0.8** | **93.1 ±0.6** | **96.3** |
| Pi0.5 (flow matching) | 10 | 96.2 ±2.0 | 96.8 ±0.4 | 92.6 ±0.3 | 86.7 ±1.4 | 93.1 |

KeyStone test-time selection lifts the Drift four-suite average to **96.5%**. Decode latency at
batch 1: **85.6 ms/chunk** (Drift) vs 259.8 ms (10-step flow matching). Expert-only training
(frozen VLM) on LIBERO-Spatial: Drift **96.3 ±0.5** vs flow matching 85.7 ±0.6.

## Install

Python >= 3.12. Pulls `lerobot[pi,dataset]>=0.6.0,<0.7`.

From GitHub:

```bash
pip install "git+https://github.com/zuoxingdong/lerobot_policy_pi05_drift.git"
```

From a local clone (editable):

```bash
git clone https://github.com/zuoxingdong/lerobot_policy_pi05_drift.git
cd lerobot_policy_pi05_drift
pip install -e .
```

The LIBERO evaluation below additionally needs:

```bash
pip install "lerobot[libero]"
pip install "mujoco==3.3.2"   # newer MuJoCo changes rendered colors
```

## Train on LIBERO

The winning drift recipe (G=8, temperatures (0.02, 0.05, 0.2), per-action-dim) is the config
default, so no drift flags are needed. The headline models warm-start the `lerobot/pi05_libero`
VLM with a freshly re-initialized action expert and finetune the full model:

```bash
lerobot-train \
  --policy.type=pi05_drift \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.fresh_action_expert=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --dataset.repo_id=lerobot/libero \
  --batch_size=56 \
  --seed=1000 \
  --steps=10000
```

For the expert-only variant (frozen VLM, ~4× cheaper, 96.3 ±0.5 on LIBERO-Spatial where flow
matching drops to 85.7 ±0.6), set
`--policy.freeze_vision_encoder=true --policy.train_expert_only=true` and `--steps=6000`.

## Evaluate on LIBERO

Needs a graphics-capable node for the MuJoCo/LIBERO renderer. The protocol is 50 episodes per
task, `n_action_steps=10`, seeds {1000, 1001, 1002}; e.g. for `libero_10` (Long):

```bash
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  lerobot-eval \
    --policy.path=<checkpoint-path-or-hub-id> \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids="[$task_id]" \
    --env.control_mode=relative \
    --env.obs_type=pixels_agent_pos \
    --env.observation_width=256 \
    --env.observation_height=256 \
    --env.init_states=true \
    --eval.n_episodes=50 \
    --eval.batch_size=2 \
    --eval.use_async_envs=true \
    --seed=1000 \
    --output_dir=eval/task_${task_id}
done
# then aggregate the ten eval_info.json files
```

`--policy.n_action_steps=10` is load-bearing: 1-NFE decoding makes per-step replanning cheap, but
committing fewer steps per chunk scores far lower on long-horizon suites (−24 pp on Long at
`n_action_steps=1`).

KeyStone test-time selection is an eval-time flag set (helps checkpoints with
selection-recoverable failures; ~zero added latency):

```bash
  --policy.test_time_samples=8 --policy.test_time_clusters=4 --policy.test_time_unimodal_tau=0.3
```

## Use from Python

```python
import lerobot_policy_pi05_drift              # registers "pi05_drift" — import before loading
from lerobot.policies.factory import get_policy_class

policy = get_policy_class("pi05_drift").from_pretrained("<checkpoint-path-or-hub-id>")
action = policy.select_action(batch)          # (B, action_dim); 1-NFE drift inference
```

The `import lerobot_policy_pi05_drift` line is required in scripts: LeRobot auto-discovers the
plugin inside the `lerobot-*` CLIs, but not in your own Python process.

## Provenance & license

Only the two files the drift objective touches are vendored — the config and the model — from
LeRobot's Pi0.5 at the released
[`lerobot==0.6.0`](https://github.com/huggingface/lerobot/releases/tag/v0.6.0), with only class
renames, the registration string, import fixes, and drift-recipe defaults on top; `drifting_util.py`
and `keystone_util.py` are new. Everything Pi0.5-Drift shares with Pi0.5 (the PaliGemma/expert
backbone module, the pre/post-processors) is imported from the installed `lerobot` directly, and
the weight layout is byte-identical — Pi0.5-format drift checkpoints load bit-for-bit (rewrite
`"type": "pi05"` to `"type": "pi05_drift"` in their `config.json` first). Apache-2.0 (`LICENSE`);
the original HuggingFace / Physical Intelligence copyright headers are retained in the vendored
files.
