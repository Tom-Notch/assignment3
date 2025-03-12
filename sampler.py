#!/usr/bin/env python3
import math
from typing import List

import torch
from pytorch3d.renderer.cameras import CamerasBase

from ray_utils import RayBundle


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth,
            self.n_pts_per_ray,
            device=ray_bundle.directions.device,
        )

        # TODO (Q1.4): Sample points from z values
        normalized_direction = ray_bundle.directions / ray_bundle.directions.norm(
            dim=-1, keepdim=True
        )

        # shape (1, num_rays x num_points_per_ray, 3)
        sample_points = (
            ray_bundle.origins
            + normalized_direction[:, :, None, :] * z_vals[None, None, :, None]
        ).reshape(1, -1, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {"stratified": StratifiedRaysampler}
