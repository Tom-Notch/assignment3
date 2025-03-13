#!/usr/bin/env python3
from typing import List
from typing import Optional
from typing import Tuple

import torch
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = (
            cfg.white_background if "white_background" in cfg else False
        )

    def _compute_weights(self, deltas, rays_density: torch.Tensor, eps: float = 1e-10):
        # TODO (1.5): Compute transmittance using the equation described in the README
        exponents = rays_density * deltas
        cumulative_exponents = torch.cumsum(exponents, dim=1)
        shifted_cumulative_exponents = torch.cat(
            [torch.zeros_like(exponents[:, :1, :]), cumulative_exponents],
            dim=1,
        )  # prepend 0 since T = 1 for the first segments

        transmittance = torch.exp(-shifted_cumulative_exponents + eps)

        alpha = 1 - torch.exp(-exponents + eps)

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = transmittance[:, :-1, :] * alpha

        return weights

    def _aggregate(self, weights: torch.Tensor, rays_feature: torch.Tensor):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start : chunk_start + self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output["density"]
            feature = implicit_output["feature"]

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1), density.view(-1, n_pts, 1)
            )

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(
                weights.view(-1, n_pts, 1), feature.view(*weights.shape[:-1], -1)
            )

            # TODO (1.5): Render depth map
            depth = self._aggregate(
                weights.view(-1, n_pts, 1), depth_values.view(*weights.shape[:-1], -1)
            )

            # Return
            cur_out = {
                "feature": feature,
                "depth": depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat([chunk_out[k] for chunk_out in chunk_outputs], dim=0)
            for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters

    def sphere_tracing(
        self,
        implicit_fn,
        origins,  # Nx3
        directions,  # Nx3
        eps: float = 1e-3,  # We'll mark rays as having converged when their SDF value is below this epsilon.
    ):
        """
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        """
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not

        device = origins.device
        N = origins.shape[0]
        # Starting distance along each ray
        t = torch.full((N,), self.near, device=device)  # (N,)
        # Boolean mask for rays that have converged (hit the surface)
        converged = torch.zeros((N,), dtype=torch.bool, device=device)

        for _ in range(self.max_iters):
            # Compute current points along each ray: (N, 3)
            points = origins + t.unsqueeze(-1) * directions
            # Evaluate SDF at these points. Expected shape is (N, 1), so squeeze last dim.
            sdf = implicit_fn(points).squeeze(-1)  # (N,)
            # Determine which active rays have reached the surface (within eps)
            hit = sdf.abs() < eps

            # Only update the mask for rays that are still active (haven't converged and t < far)
            active = (t < self.far) & (~converged)
            # For active rays, if they now satisfy the hit condition, mark them as converged.
            converged = converged | (active & hit)
            # For active rays that haven't hit, update t by marching along the ray.
            t = torch.where(active & (~hit), t + sdf, t)
            # If no ray is still active, we can break early.
            if (~(converged | (t >= self.far))).sum() == 0:
                break

        # Final estimated intersection points.
        final_points = origins + t.unsqueeze(-1) * directions
        # Mask has shape (N, 1)
        mask = converged.unsqueeze(-1)
        return final_points, mask

    def forward(self, sampler, implicit_fn, ray_bundle, light_dir=None):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start : chunk_start + self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn, cur_ray_bundle.origins, cur_ray_bundle.directions
            )
            mask = mask.repeat(1, 3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                "color": color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat([chunk_out[k] for chunk_out in chunk_outputs], dim=0)
            for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(
    signed_distance: torch.Tensor,
    alpha: float,
    beta: float,
):
    """
    Converts a signed distance value to volume density using the VolSDF formulation.

    For points with signed_distance > 0 (outside the surface):
      density = alpha * 0.5 * exp(-signed_distance / beta)
    For points with signed_distance <= 0 (inside the surface):
      density = alpha * (1 - 0.5 * exp(signed_distance / beta))

    Args:
        signed_distance: Tensor of SDF values.
        alpha: Learnable scaling factor.
        beta: Learnable parameter controlling the sharpness.
        eps: A small constant for numerical stability.

    Returns:
        density: Tensor of volume densities.
    """
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    s = -signed_distance
    density = alpha * torch.where(
        s <= 0,
        0.5 * torch.exp(s / beta),
        1 - 0.5 * torch.exp(-s / beta),
    )
    return density


class VolumeSDFRenderer(VolumeRenderer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = (
            cfg.white_background if "white_background" in cfg else False
        )
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(self, sampler, implicit_fn, ray_bundle, light_dir=None):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start : chunk_start + self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(
                cur_ray_bundle.sample_points
            )
            density = sdf_to_density(
                distance, self.alpha, self.beta
            )  # TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1), density.view(-1, n_pts, 1)
            )

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(weights, color.view(-1, n_pts, color.shape[-1]))

            # Return
            cur_out = {"color": color, "geometry": geometry_color}

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat([chunk_out[k] for chunk_out in chunk_outputs], dim=0)
            for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    "volume": VolumeRenderer,
    "sphere_tracing": SphereTracingRenderer,
    "volume_sdf": VolumeSDFRenderer,
}
