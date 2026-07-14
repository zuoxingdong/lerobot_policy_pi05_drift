#!/usr/bin/env python

# Copyright 2026 Xingdong Zuo. All rights reserved.
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

"""Drifting loss (torch-only) for the one-step "Drifting" (DBPO) objective.

Faithful structural port of the official JAX `drift_loss` (lambertae/drifting,
`drift_loss.py`) restricted to the surface this integration uses: the model's
own samples serve as the negatives (sibling self-repulsion), so the API takes
only `gen` and `data`. The reference's explicit-negative and per-sample-weight
plumbing is deliberately not carried -- with weights of ones and no explicit
negatives it is numerically the identity, and re-adding it is one concat away.

Layout convention: an optional leading GROUP dim batches independent matching
problems that must NOT share statistics. Within one group, the distance ruler
`scale` and each temperature's force RMS are global scalars; everything else is
local to a single (group, b) matching problem. The policy integration uses
groups = chunk timesteps (per-timestep mode), REAL action dims (per-dim mode),
or one group over the flattened chunk (flat mode), which makes the
granularities a single code path.

Gradient contract: the drifted goal is frozen (computed under `no_grad`, with
AMP autocast force-disabled -- see `drift_loss_grouped`); gradient flows only
through the live `gen` in the final residual, so minimizing the loss moves each
sample along its own field: `mse(gen, stopgrad(gen + V))`.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812


def validate_temperatures(temperatures):
    """Return `temperatures` as a tuple of finite positive floats, or raise."""
    try:
        temps = tuple(float(r) for r in temperatures)
    except TypeError as exc:
        raise ValueError("`temperatures` must be a non-empty iterable of finite positive floats.") from exc
    if len(temps) == 0:
        raise ValueError("`temperatures` must contain at least one positive value.")
    for r in temps:
        if not math.isfinite(r) or r <= 0:
            raise ValueError(f"`temperatures` must contain only finite positive values; got {temps}.")
    return temps


def _pairwise_dist(x, y, eps=1e-8):
    """Pairwise L2 distance, [..., N, D] x [..., M, D] -> [..., N, M].

    Exact port of the official JAX cdist (dot-product formula + eps clamp);
    matches the reference kernel rather than `torch.cdist`.
    """
    xy = torch.einsum("...nd,...md->...nm", x, y)
    xx = torch.einsum("...nd,...nd->...n", x, x)
    yy = torch.einsum("...md,...md->...m", y, y)
    sq_dist = xx.unsqueeze(-1) + yy.unsqueeze(-2) - 2 * xy
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def _drift_goal(gen_detached, data, valid, temps):
    """Drifted targets for a batch of groups. fp32 inputs, runs under no_grad.

    gen_detached: [T, B, C, S] detached samples; data: [T, B, P, S] positives;
    valid: [T, B] float mask; temps: tuple of K temperatures.

    Returns (goal [T, B, C, S], scale_inputs [T], scale [T], energy [K, T]).
    Group statistics (scale, per-temperature force RMS) are masked means over
    each group's valid units. Empty groups get placeholder statistics (scale
    1.0, energy 0.0) and produce garbage-but-finite goal rows; callers exclude
    them via `valid` (their loss units are zeroed).
    """
    c_gen, s_dim = gen_detached.shape[2], gen_detached.shape[3]
    cand = torch.cat([gen_detached, data], dim=2)  # [T, B, Y, S]; siblings first
    y_cand = cand.shape[2]
    v_unit = valid[:, :, None, None]

    # Stage 1: one data-dependent ruler per group (masked mean of all pair
    # distances), then express samples and distances in units of it.
    dist = _pairwise_dist(gen_detached, cand)  # [T, B, C, Y]
    pair_count = valid.sum(dim=1) * (c_gen * y_cand)  # [T]
    scale = (dist * v_unit).sum(dim=(1, 2, 3)) / pair_count.clamp(min=1.0)
    scale = torch.where(pair_count > 0, scale, torch.ones_like(scale))
    scale_inputs = torch.clamp(scale / (s_dim**0.5), min=1e-3)

    gen_scaled = gen_detached / scale_inputs[:, None, None, None]
    cand_scaled = cand / scale_inputs[:, None, None, None]
    dist_normed = dist / torch.clamp(scale, min=1e-3)[:, None, None, None]

    # Mask gen self-connections (added to the NORMALIZED distance, so the
    # masking strength scales with the temperature).
    self_mask = F.pad(torch.eye(c_gen, device=dist.device, dtype=dist.dtype), (0, y_cand - c_gen))
    dist_normed = dist_normed + self_mask * 100.0

    # Stages 2-3, all temperatures at once. [K, T, B, C, Y]:
    # bistochastic affinity (geometric mean of row- and column-softmax), split
    # into the sibling block and the data block, mass-coupled so coefficient
    # rows sum to ~0 and the force is a pure sum of (y - g) displacements.
    r_view = dist_normed.new_tensor(temps).view(-1, 1, 1, 1, 1)
    logits = -dist_normed.unsqueeze(0) / r_view
    aff = torch.sqrt(torch.clamp(logits.softmax(dim=-1) * logits.softmax(dim=-2), min=1e-6))
    aff_sib, aff_data = aff[..., :c_gen], aff[..., c_gen:]
    coeff = torch.cat(
        [-aff_sib * aff_data.sum(dim=-1, keepdim=True), aff_data * aff_sib.sum(dim=-1, keepdim=True)],
        dim=-1,
    )

    # Stage 4: per-(temperature, group) unit-RMS forces, summed over K.
    force = torch.einsum("ktbcy,tbys->ktbcs", coeff, cand_scaled)
    force = force - coeff.sum(dim=-1, keepdim=True) * gen_scaled
    unit_count = valid.sum(dim=1) * (c_gen * s_dim)  # [T]
    energy = (force.square() * v_unit).sum(dim=(2, 3, 4)) / unit_count.clamp(min=1.0)  # [K, T]
    force_rms = torch.sqrt(torch.clamp(energy, min=1e-8))
    drift = (force / force_rms[:, :, None, None, None]).sum(dim=0)  # [T, B, C, S]

    return gen_scaled + drift, scale_inputs, scale, energy


def drift_loss_grouped(gen, data, valid=None, temperatures=(0.02, 0.05, 0.2)):
    """Drift loss over a leading group dim; groups have independent statistics.

    Args:
        gen: [T, B, C, S] generated samples (carry gradient). The C siblings of
            each (group, b) matching problem serve as each other's negatives.
        data: [T, B, P, S] data points (the positives), P >= 1.
        valid: optional [T, B] bool. Invalid units are excluded from group
            statistics and contribute exactly zero loss and zero gradient.
        temperatures: bandwidths; each contributes one unit-RMS force.

    Returns:
        loss: [T, B] per-unit drift energy, exactly zero at invalid units.
        info: detached per-group tensors {"scale": [T], "loss_{R}": [T]}.
            Empty groups hold placeholders; average with `valid.any(dim=1)`.
    """
    temps = validate_temperatures(temperatures)
    # At-least-fp32 compute dtype: bf16/fp16 inputs are promoted to fp32; float64
    # inputs (verification/arbitration runs) keep full precision.
    compute_dtype = torch.float64 if gen.dtype == torch.float64 else torch.float32
    gen = gen.to(compute_dtype)
    data = data.to(compute_dtype)
    v = gen.new_ones(gen.shape[:2]) if valid is None else valid.to(gen.dtype)

    # Goal computation (no gradients) -- the drifted target is frozen. Force
    # at-least-fp32 even under AMP autocast: the cdist/softmax/einsum chain is
    # numerically delicate (bf16 shifts per-sample losses by ~1e-1) and the
    # official JAX reference runs it in fp32. The dtype promotion above does
    # not survive autocast on its own (einsum inputs are re-cast on entry),
    # hence the explicit disable.
    with torch.no_grad(), torch.autocast(device_type=gen.device.type, enabled=False):
        goal, scale_inputs, scale, energy = _drift_goal(gen.detach(), data, v, temps)

    # Loss with gradients through gen only (everything above is grad-free).
    # Masking the residual BEFORE squaring keeps invalid units at exactly zero
    # value and zero gradient even if their (excluded) goal rows are extreme.
    gen_scaled = gen / scale_inputs[:, None, None, None]
    diff = (gen_scaled - goal) * v[:, :, None, None]
    loss = diff.square().mean(dim=(-1, -2))  # [T, B]

    info = {"scale": scale}
    for r, e in zip(temps, energy, strict=True):
        info[f"loss_{r}"] = e
    return loss, info


def drift_loss(gen, data, temperatures=(0.02, 0.05, 0.2)):
    """Single-group drift loss: gen [B, C, S], data [B, P, S] -> ([B], info).

    Thin wrapper over `drift_loss_grouped` with one group; info values are
    0-dim detached tensors ("scale" and one "loss_{R}" per temperature).
    """
    loss, info = drift_loss_grouped(gen.unsqueeze(0), data.unsqueeze(0), None, temperatures)
    return loss.squeeze(0), {k: t.squeeze(0) for k, t in info.items()}
