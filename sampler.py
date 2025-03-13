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
        )[
            None, :, None
        ]  # shape(1, num_points_per_ray, 1)

        # TODO (Q1.4): Sample points from z values
        # shape (num_rays, num_points_per_ray, 3)
        sample_points = (
            ray_bundle.origins[:, None, :] + ray_bundle.directions[:, None, :] * z_vals
        )

        # Return
        return RayBundle(
            origins=ray_bundle.origins,
            directions=ray_bundle.directions,
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {"stratified": StratifiedRaysampler}
