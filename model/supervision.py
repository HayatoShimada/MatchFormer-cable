# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
#
# This file re-implements the coarse/fine ground-truth supervision pattern
# from LoFTR (https://github.com/zju3dv/LoFTR, Apache-2.0), specifically
# ``src/loftr/utils/supervision.py``. New file in this fork.
"""Coarse + fine ground-truth supervision for MatchFormer-cable training.

Ported from LoFTR's ``src/loftr/utils/supervision.py``. Given a batch with
``depth0/1``, ``K0/1``, ``T_0to1/1to0`` and the coarse/fine grid sizes computed
by the network's forward pass, this fills ``data`` with:

  * ``conf_matrix_gt``     [N, L, S]  bool  GT coarse correspondence matrix
  * ``spv_b_ids/i_ids/j_ids``           index lists used by CoarseMatching's
                                        training-time padding (see
                                        third_party/MatchFormer/model/backbone/
                                        coarse_matching.py:167+).
  * ``spv_w_pt0_i``        [N, hw0, 2] image-1 coords of every coarse cell on
                                        image-0, in image-1 pixel space (used
                                        by the fine supervision).
  * ``spv_pt1_i``          [N, hw1, 2] image-1 coords for every coarse cell on
                                        image-1 (used to compute the fine GT).
  * ``expec_f_gt``         [M, 2]      GT sub-pixel offset within each fine
                                        window for each selected match.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Project keypoints from image-0 to image-1 using depth + relative pose.

    Args:
        kpts0    [N, L, 2]  pixel coords (x, y) on image-0.
        depth0   [N, H0, W0]
        depth1   [N, H1, W1]
        T_0to1   [N, 4, 4]
        K0, K1   [N, 3, 3]
    Returns:
        valid_mask [N, L]   bool  -- depth>0 + in-bounds + depth-consistent
        warped_kpts0 [N, L, 2]    (x, y) coords on image-1 (zeros where invalid)
    """
    kpts0_long = kpts0.round().long()

    # 1. sample depth at kpts0
    n, l, _ = kpts0.shape
    h0, w0 = depth0.shape[-2:]
    x = kpts0_long[..., 0].clamp(0, w0 - 1)
    y = kpts0_long[..., 1].clamp(0, h0 - 1)
    kpts0_depth = torch.stack(
        [depth0[i, y[i], x[i]] for i in range(n)], dim=0)  # [N, L]
    nonzero_mask = kpts0_depth > 0

    # 2. unproject to camera-0 frame
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[..., :1])], dim=-1) * kpts0_depth[..., None]  # [N, L, 3]
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # [N, 3, L]

    # 3. transform to camera-1 frame
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # [N, 3, L]
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # 4. project to image-1
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # [N, L, 3]
    w_kpts0 = w_kpts0_h[..., :2] / (w_kpts0_h[..., [2]] + 1e-4)  # [N, L, 2]

    h1, w1 = depth1.shape[-2:]
    covisible_mask = (
        (w_kpts0[..., 0] > 0) & (w_kpts0[..., 0] < w1 - 1)
        & (w_kpts0[..., 1] > 0) & (w_kpts0[..., 1] < h1 - 1)
    )

    # 5. depth consistency check
    w_kpts0_long = w_kpts0.round().long()
    w_kpts0_long[..., 0] = w_kpts0_long[..., 0].clamp(0, w1 - 1)
    w_kpts0_long[..., 1] = w_kpts0_long[..., 1].clamp(0, h1 - 1)
    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(n)],
        dim=0,
    )  # [N, L]
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed).abs()
                       / w_kpts0_depth.clamp(min=1e-4)) < 0.2
    valid_mask = nonzero_mask & covisible_mask & consistent_mask
    return valid_mask, w_kpts0


# --------------------------------------------------------------------------- #
#  Coarse supervision
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_supervision_coarse(data, scale_c=8):
    """Compute coarse-level GT matches and write them into ``data``.

    Mirrors LoFTR's ``compute_supervision_coarse``. Assumes the caller has
    already run a forward pass so ``data['hw0_c']`` etc. exist; in this
    pipeline we instead supply ``hw0_c/hw1_c`` from the data module before
    the forward pass (see lightning_module.py).
    """
    device = data["image0"].device
    n, _, h0, w0 = data["image0"].shape
    _, _, h1, w1 = data["image1"].shape
    h0c, w0c = h0 // scale_c, w0 // scale_c
    h1c, w1c = h1 // scale_c, w1 // scale_c

    # 1. coarse-grid pixel coordinates (cell centres at (i+0.5)*scale)
    grid_pt0_c = create_meshgrid(h0c, w0c, False, device).reshape(1, -1, 2).repeat(n, 1, 1)  # [N, L, 2]
    grid_pt0_i = scale_c * grid_pt0_c  # in original-resolution pixels
    grid_pt1_c = create_meshgrid(h1c, w1c, False, device).reshape(1, -1, 2).repeat(n, 1, 1)
    grid_pt1_i = scale_c * grid_pt1_c

    # 2. project both grids both ways
    valid_mask0, w_pt0_i = warp_kpts(
        grid_pt0_i, data["depth0"], data["depth1"], data["T_0to1"], data["K0"], data["K1"])
    valid_mask1, w_pt1_i = warp_kpts(
        grid_pt1_i, data["depth1"], data["depth0"], data["T_1to0"], data["K1"], data["K0"])

    # 3. nearest cell on the *other* image
    w_pt0_c = w_pt0_i / scale_c   # cell coords on image-1
    w_pt1_c = w_pt1_i / scale_c
    nearest_index1 = (w_pt0_c[..., 0].round().long().clamp(0, w1c - 1)
                      + w_pt0_c[..., 1].round().long().clamp(0, h1c - 1) * w1c)  # [N, L]
    nearest_index0 = (w_pt1_c[..., 0].round().long().clamp(0, w0c - 1)
                      + w_pt1_c[..., 1].round().long().clamp(0, h0c - 1) * w0c)  # [N, S]

    # 4. mutual nearest = valid + cycle-consistent
    loop_back = torch.stack(
        [nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)])  # [N, L]
    correct_0to1 = loop_back == torch.arange(h0c * w0c, device=device).expand(n, -1)
    correct_0to1[:, 0] = False  # remove (0,0)<->(0,0) artefacts on padded inputs
    valid = valid_mask0 & correct_0to1

    # 5. assemble GT conf matrix and index lists
    conf_matrix_gt = torch.zeros(n, h0c * w0c, h1c * w1c, dtype=torch.bool, device=device)
    b_ids, i_ids = torch.where(valid)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = True

    data.update({
        "conf_matrix_gt": conf_matrix_gt,
        "spv_b_ids": b_ids,
        "spv_i_ids": i_ids,
        "spv_j_ids": j_ids,
        "spv_w_pt0_i": w_pt0_i,   # image-1 pixel coords for every coarse-grid cell on image-0
        "spv_pt1_i": grid_pt1_i,  # image-1 pixel coords for every coarse-grid cell on image-1
        "hw0_c": (h0c, w0c),
        "hw1_c": (h1c, w1c),
    })


# --------------------------------------------------------------------------- #
#  Fine supervision
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_supervision_fine(data, fine_window_size=5, scale_f=2):
    """GT sub-pixel offsets inside each fine window for the selected matches.

    Must be called *after* the forward pass: it consumes ``b_ids/i_ids/j_ids``
    written by CoarseMatching and writes ``expec_f_gt`` for the L2 fine loss.
    """
    b_ids = data["b_ids"]
    if b_ids.numel() == 0:
        data["expec_f_gt"] = torch.empty(0, 2, device=data["image0"].device)
        return

    # 1. coords of selected i_ids on image-0 (coarse grid) -> warp to image-1
    w_pt0_i = data["spv_w_pt0_i"][b_ids, data["i_ids"]]   # [M, 2]  image-1 pixel coords

    # 2. coords of selected j_ids on image-1 (coarse grid)
    pt1_i = data["spv_pt1_i"][b_ids, data["j_ids"]]       # [M, 2]

    # 3. residual in image-1 fine-resolution pixels, normalised to [-1, 1] over
    #    the fine window (size W = fine_window_size at fine resolution = scale_f).
    half_w = (fine_window_size // 2) * scale_f
    expec_f_gt = (w_pt0_i - pt1_i) / half_w  # [M, 2]
    data["expec_f_gt"] = expec_f_gt


# --------------------------------------------------------------------------- #
#  small utility (kornia's create_meshgrid clone with explicit (x, y) ordering)
# --------------------------------------------------------------------------- #

def create_meshgrid(h: int, w: int, normalised: bool, device) -> torch.Tensor:
    if normalised:
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
    else:
        ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
