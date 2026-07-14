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

"""KeyStone-style test-time self-consistency selection over K sampled action chunks.

Torch port of the judge-free selector from "Geometry Guided Self-Consistency for
Physical AI" (Dai et al., 2026, arXiv:2605.08638), Section 3.3. Given K candidate
chunks drawn from ONE shared observation context, pick the medoid of the dominant
cluster in flattened action space:

  1. pairwise L2 distances Delta_ij over the K flattened chunks;
  2. unimodality guard: s = ||mean - global_medoid|| / (median_{i<j} Delta_ij + eps);
     s < tau -> the candidates are one compact cluster, return the GLOBAL medoid
     (k-means on a unimodal batch would split it into arbitrary halves);
  3. otherwise k-means (C clusters, centroids initialized from C distinct random
     candidates, <= 10 iterations, early stop on stable assignments), then the
     medoid of the LARGEST cluster.

The selected chunk is always one of the model's own K outputs -- never an average
-- so it stays on the sampled action manifold and cannot interpolate between
distinct modes. No learned parameters, no auxiliary scorer.

Everything here is pure tensor math over [B, K, F] flattened candidates; the caller
decides what F is (slice to the REAL action dims before flattening, so padding
columns never vote).
"""

import torch
from torch import Tensor

_EPS = 1e-8


@torch.no_grad()
def cluster_medoid_select(
    candidates: Tensor,
    num_clusters: int = 2,
    unimodal_tau: float = 0.3,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict]:
    """Pick one candidate per batch element by guarded cluster-medoid selection.

    Args:
        candidates: [B, K, F] flattened candidate chunks (any float dtype; distance
            math runs in fp32). K == 1 degenerates to index 0 everywhere.
        num_clusters: C for the k-means stage (clamped to K).
        unimodal_tau: guard threshold; below it the global medoid is returned.
        generator: optional torch.Generator for the k-means centroid init draw
            (tests pin it for determinism; deployment leaves it None).

    Returns:
        (indices, info): indices [B] long -- the selected candidate per batch
        element; info holds per-batch diagnostics (spread statistic, whether the
        guard fired, the winning cluster's size) as CPU lists.
    """
    if candidates.ndim != 3:
        raise ValueError(f"candidates must be [B, K, F], got shape {tuple(candidates.shape)}")
    bsize, k, _ = candidates.shape
    device = candidates.device
    if k == 1:
        zeros = torch.zeros(bsize, dtype=torch.long, device=device)
        return zeros, {"spread": [0.0] * bsize, "unimodal": [True] * bsize, "cluster_size": [1] * bsize}

    x = candidates.float()
    dist = torch.cdist(x, x)  # [B, K, K]

    indices = torch.zeros(bsize, dtype=torch.long, device=device)
    spread_out, unimodal_out, size_out = [], [], []
    triu_i, triu_j = torch.triu_indices(k, k, offset=1, device=device)
    for b in range(bsize):
        d = dist[b]
        medoid = int(d.sum(dim=1).argmin())
        # Unimodality guard (paper eq. 4): distance of the sample mean from the
        # global medoid, normalized by the median pairwise distance.
        med_pair = d[triu_i, triu_j].median()
        spread = float((x[b].mean(dim=0) - x[b, medoid]).norm() / (med_pair + _EPS))
        spread_out.append(spread)
        if spread < unimodal_tau:
            indices[b] = medoid
            unimodal_out.append(True)
            size_out.append(k)
            continue
        unimodal_out.append(False)
        assign = _kmeans_assign(x[b], min(num_clusters, k), generator)
        counts = torch.bincount(assign, minlength=int(assign.max()) + 1)
        members = (assign == int(counts.argmax())).nonzero(as_tuple=True)[0]
        # Medoid WITHIN the winning cluster, reusing the precomputed distances.
        within = d[members][:, members].sum(dim=1)
        indices[b] = members[int(within.argmin())]
        size_out.append(int(counts.max()))

    return indices, {"spread": spread_out, "unimodal": unimodal_out, "cluster_size": size_out}


def _kmeans_assign(x: Tensor, num_clusters: int, generator: torch.Generator | None) -> Tensor:
    """Small k-means over [K, F]: centroids from C distinct random candidates,
    <= 10 Lloyd iterations, early stop when assignments stabilize. Returns [K] long."""
    k = x.shape[0]
    # Draw the init permutation on CPU (torch.Generator defaults to CPU; a CUDA
    # randperm with a CPU generator raises), then move the indices to x's device.
    init = torch.randperm(k, generator=generator)[:num_clusters].to(x.device)
    centroids = x[init].clone()
    assign: Tensor | None = None
    for _ in range(10):
        new_assign = torch.cdist(x, centroids).argmin(dim=1)
        if assign is not None and torch.equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(num_clusters):
            mask = assign == c
            if mask.any():  # empty cluster keeps its old centroid
                centroids[c] = x[mask].mean(dim=0)
    return assign
