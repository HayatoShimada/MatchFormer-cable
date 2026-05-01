# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
# New file in MatchFormer-cable fork (no upstream counterpart).
"""Coarse + fine GT supervision for homography-paired training.

Counterpart of ``model/supervision.py`` (which uses depth + relative pose).
Used when the dataset synthesises pairs by warping a single masked frame with
a known homography ``H``: image1 = H(image0). Since the warp is exact, GT
correspondences are computable analytically:

  * Every cell on the image-0 coarse grid maps to ``H @ cell`` in image-1 px.
  * The valid mask is "warped pixel lies inside image-1 (and inside the
    image-1 mask if provided)".

Inputs in ``data`` (besides image0/1, mask0_full/mask1_full):
  * data['H_0to1']   [N, 3, 3]  homography mapping image-0 px -> image-1 px

Outputs (same keys as supervision.py so the loss code is shared):
  * conf_matrix_gt
  * spv_b_ids / spv_i_ids / spv_j_ids
  * spv_w_pt0_i      [N, hw0_c, 2]  image-1 px coords for each coarse cell on image-0
  * spv_pt1_i        [N, hw1_c, 2]  image-1 px coords for each coarse cell on image-1
  * expec_f_gt       [M, 2]
"""

from __future__ import annotations

import torch

from .supervision import create_meshgrid


@torch.no_grad()
def _warp_h(pts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Apply 3x3 H to (x, y) points. pts: [N, L, 2], H: [N, 3, 3] -> [N, L, 2]."""
    ones = torch.ones_like(pts[..., :1])
    p_h = torch.cat([pts, ones], dim=-1)  # [N, L, 3]
    out = torch.einsum("nij,nlj->nli", H, p_h)
    return out[..., :2] / (out[..., 2:3] + 1e-8)


@torch.no_grad()
def compute_supervision_coarse_h(data: dict, scale_c: int = 8) -> None:
    device = data["image0"].device
    n, _, h0, w0 = data["image0"].shape
    _, _, h1, w1 = data["image1"].shape
    h0c, w0c = h0 // scale_c, w0 // scale_c
    h1c, w1c = h1 // scale_c, w1 // scale_c

    # 1. coarse-grid pixel coords (cell centres at (i+0.5)*scale)
    grid_pt0_c = create_meshgrid(h0c, w0c, False, device).reshape(1, -1, 2).repeat(n, 1, 1)
    grid_pt0_i = scale_c * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1c, w1c, False, device).reshape(1, -1, 2).repeat(n, 1, 1)
    grid_pt1_i = scale_c * grid_pt1_c

    H = data["H_0to1"].to(device).float()
    H_inv = torch.linalg.inv(H)

    # 2. project both grids both ways
    w_pt0_i = _warp_h(grid_pt0_i, H)        # image-0 cells in image-1 px
    w_pt1_i = _warp_h(grid_pt1_i, H_inv)    # image-1 cells in image-0 px

    # 3. validity (in-bounds + optional mask gating)
    in_bounds_0 = ((w_pt0_i[..., 0] >= 0) & (w_pt0_i[..., 0] < w1)
                   & (w_pt0_i[..., 1] >= 0) & (w_pt0_i[..., 1] < h1))
    in_bounds_1 = ((w_pt1_i[..., 0] >= 0) & (w_pt1_i[..., 0] < w0)
                   & (w_pt1_i[..., 1] >= 0) & (w_pt1_i[..., 1] < h0))

    if "mask0_full" in data and data["mask0_full"] is not None:
        mask0_full = data["mask0_full"].to(device).bool()
        x0 = grid_pt0_i[..., 0].long().clamp(0, w0 - 1)
        y0 = grid_pt0_i[..., 1].long().clamp(0, h0 - 1)
        valid_mask_a = torch.stack([mask0_full[i, y0[i], x0[i]] for i in range(n)], dim=0)
        in_bounds_0 = in_bounds_0 & valid_mask_a

    if "mask1_full" in data and data["mask1_full"] is not None:
        mask1_full = data["mask1_full"].to(device).bool()
        wx = w_pt0_i[..., 0].round().long().clamp(0, w1 - 1)
        wy = w_pt0_i[..., 1].round().long().clamp(0, h1 - 1)
        valid_mask_b = torch.stack([mask1_full[i, wy[i], wx[i]] for i in range(n)], dim=0)
        in_bounds_0 = in_bounds_0 & valid_mask_b

    # 4. nearest cell index on image-1 for each image-0 cell
    w_pt0_c = w_pt0_i / scale_c
    nearest_index1 = (w_pt0_c[..., 0].round().long().clamp(0, w1c - 1)
                      + w_pt0_c[..., 1].round().long().clamp(0, h1c - 1) * w1c)

    # 5. NO cycle-consistency filter: the homography is bijective by
    #    construction, so the analytical projection IS the GT. The strict
    #    round-trip check used in the depth+pose path rejects cells that
    #    cross-cell-boundaries under even tiny rotations -- inappropriate
    #    here. We still de-duplicate j_ids (multiple i_ids landing on the
    #    same image-1 cell): keep only the i_id closest to the cell centre.
    valid = in_bounds_0

    # 6. assemble GT, deduplicating per (batch, j) via the smallest residual
    #    distance to the cell centre (in image-1 px space).
    conf_matrix_gt = torch.zeros(n, h0c * w0c, h1c * w1c, dtype=torch.bool, device=device)
    b_ids_all, i_ids_all = torch.where(valid)
    j_ids_all = nearest_index1[b_ids_all, i_ids_all]

    # Distance from warped pt to the centre of the j-cell, used as a tiebreak.
    j_cell_x = (j_ids_all % w1c).float() * scale_c + 0.5 * scale_c
    j_cell_y = (j_ids_all // w1c).float() * scale_c + 0.5 * scale_c
    warp_xy = w_pt0_i[b_ids_all, i_ids_all]
    dist = (warp_xy[..., 0] - j_cell_x) ** 2 + (warp_xy[..., 1] - j_cell_y) ** 2

    keep = torch.ones_like(b_ids_all, dtype=torch.bool)
    if b_ids_all.numel():
        # Per (batch, j) keep the smallest-dist i.
        flat_key = b_ids_all * (h1c * w1c) + j_ids_all
        order = torch.argsort(dist)
        seen = torch.zeros(n * h1c * w1c, dtype=torch.bool, device=device)
        keep_re = torch.zeros_like(keep)
        for idx in order.tolist():
            k = flat_key[idx].item()
            if not seen[k]:
                seen[k] = True
                keep_re[idx] = True
        keep = keep_re

    b_ids = b_ids_all[keep]
    i_ids = i_ids_all[keep]
    j_ids = j_ids_all[keep]
    conf_matrix_gt[b_ids, i_ids, j_ids] = True

    data.update({
        "conf_matrix_gt": conf_matrix_gt,
        "spv_b_ids": b_ids,
        "spv_i_ids": i_ids,
        "spv_j_ids": j_ids,
        "spv_w_pt0_i": w_pt0_i,
        "spv_pt1_i": grid_pt1_i,
        "hw0_c": (h0c, w0c),
        "hw1_c": (h1c, w1c),
    })


@torch.no_grad()
def compute_supervision_fine_h(data: dict, fine_window_size: int = 5,
                               scale_f: int = 2) -> None:
    """Identical to supervision.compute_supervision_fine -- the residual is
    just (warped_pt0 - pt1) regardless of how the warp was produced.
    """
    b_ids = data["b_ids"]
    if b_ids.numel() == 0:
        data["expec_f_gt"] = torch.empty(0, 2, device=data["image0"].device)
        return

    w_pt0_i = data["spv_w_pt0_i"][b_ids, data["i_ids"]]
    pt1_i = data["spv_pt1_i"][b_ids, data["j_ids"]]

    half_w = (fine_window_size // 2) * scale_f
    expec_f_gt = (w_pt0_i - pt1_i) / half_w
    data["expec_f_gt"] = expec_f_gt
