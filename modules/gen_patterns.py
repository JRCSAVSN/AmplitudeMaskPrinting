import torch
import numpy as np

from .utils import *

# def gen_radial(mask_shape: tuple, n_rays:int, angle_ray:float = None,
#                 white_pixel_ratio:float = None,
#                 mask_rotation_angle:float = 0.0, acyclic:bool = False,
#                 ):
#     """
#     Generates a radial mask with n_rays, with each ray having an opening of angle_ray degrees.
#     Alternatively, specify white_pixel_ratio (in percent, e.g., 50.0) to override angle_ray.
#
#     Args:
#         mask_shape: Tuple of shape (H, W, C) or (B, H, W, C)
#         n_rays: Number of white rays in the mask
#         angle_ray: Angle of opening for the white rays (degrees, half-angle)
#         white_pixel_ratio: Desired white pixel ratio in percent (0-100)
#     """
#     # If white_pixel_ratio is provided, compute angle_ray accordingly
#     if white_pixel_ratio is not None:
#         angle_ray = (360.0 / (2 * n_rays)) * (white_pixel_ratio / 100.0)
#
#     mask_shape = (mask_shape, mask_shape)
#     # Create blank mask (1 = white, 0 = black)
#     mask = torch.ones(mask_shape, dtype=torch.bool)
#
#     # Center
#     xc = mask_shape[0]//2
#     yc = mask_shape[1]//2
#
#     # Define center line of rays
#     angle_center = 360.0 / n_rays
#     alphas = [((i*angle_center)+mask_rotation_angle)%360 for i in range(n_rays)]
#     if acyclic:
#         alphas = [alpha for alpha in alphas if random() <= 0.8]
#
#     ray_side_height = mask_shape[0] + mask_shape[1]
#
#     # Describe rays as triangles
#     triangles = []
#     for alpha in alphas:
#         x1 = ray_side_height * cos(radians(alpha) + radians(angle_ray))
#         y1 = ray_side_height * sin(radians(alpha) + radians(angle_ray))
#         x2 = ray_side_height * cos(radians(alpha) - radians(angle_ray))
#         y2 = ray_side_height * sin(radians(alpha) - radians(angle_ray))
#         triangles.append([(0, 0), (x1, y1), (x2, y2)])
#
#     # Fill pixels
#     for x in range(mask_shape[0]):
#         for y in range(mask_shape[1]):
#             for triangle in triangles:
#                 a =  (det(x-xc, y-yc, triangle[2][0], triangle[2][1]) - det(triangle[0][0], triangle[0][1], triangle[2][0], triangle[2][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
#                 b = -(det(x-xc, y-yc, triangle[1][0], triangle[1][1]) - det(triangle[0][0], triangle[0][1], triangle[1][0], triangle[1][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
#                 if a > 0 and b > 0 and a+b < 1:
#                     mask[x, y] = False
#                     break
#     return mask.numpy()

def gen_radial(H, W, n_rays, white_pixel_ratio=50.0, rotation_deg=0.0,
                             tile_h=4096, device="cpu", out_dtype=torch.bool):
    """
    Returns a torch tensor mask of shape (H, W) with values {0,1} as out_dtype.
    Processes in tiles to control memory. If device='cuda', uses GPU for speed.
    """
    angle_center = 360.0 / n_rays
    white_width  = 360.0 * (white_pixel_ratio / 100.0) / n_rays
    half_white   = white_width / 2.0

    # center (choose consistent indexing so that x=cols (W), y=rows (H))
    yc = (H - 1) / 2.0
    xc = (W - 1) / 2.0

    # Precompute x-coordinates once per row-tile to save work
    x_coords = torch.arange(W, dtype=torch.float32, device=device) - xc

    out = torch.empty((H, W), dtype=out_dtype, device="cpu")  # final on CPU

    for y0 in range(0, H, tile_h):
        h = min(tile_h, H - y0)
        # y grid for this tile
        y = torch.arange(y0, y0 + h, dtype=torch.float32, device=device) - yc
        # 2D grid via outer: shape (h, W)
        # atan2 takes (y, x)
        Y = y[:, None]            # (h,1)
        X = x_coords[None, :]     # (1,W)

        theta = torch.rad2deg(torch.atan2(Y, X))  # [-180, 180]
        # shift/rotate and fold into sector
        phase = (theta - rotation_deg) % angle_center
        dist  = torch.minimum(phase, angle_center - phase)

        white = (dist <= half_white)  # bool
        # Convert to desired uint8 binary (1=white, 0=black) or flip if you prefer
        tile_mask = white.to(out_dtype)

        # move to CPU and place into output
        out[y0:y0+h, :] = tile_mask.detach().to("cpu")

        # free tile tensors (esp. on CUDA)
        del Y, X, theta, phase, dist, white, tile_mask
        if device == "cuda":
            torch.cuda.empty_cache()

    return out.numpy()