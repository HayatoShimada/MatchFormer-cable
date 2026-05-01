# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../../LICENSE.
#
# Re-implements LoFTR's training-time loss
# (https://github.com/zju3dv/LoFTR, Apache-2.0), specifically
# ``src/loftr/utils/loftr_loss.py``. New file in this fork.
"""Coarse focal loss + fine L2 loss for MatchFormer-cable training."""

from __future__ import annotations

import torch
import torch.nn as nn


class MatchformerLoss(nn.Module):
    """Loss = w_c * focal(coarse) + w_f * L2(fine).

    Expects, after one forward pass + ``compute_supervision_coarse/fine``:

      * data['conf_matrix']    [N, L, S]  predicted dual-softmax confidence
      * data['conf_matrix_gt'] [N, L, S]  bool GT (1 where positive match)
      * data['expec_f']        [M, 3]     (dx, dy, std) per selected match
      * data['expec_f_gt']     [M, 2]     (dx, dy) GT in [-1, 1] window-norm
      * data['mask0' / 'mask1'] (optional) coarse-level token validity masks;
        when present, the focal loss is restricted to (mask0 ⊗ mask1) cells so
        background tokens never produce gradients.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.coarse_w = float(config.get("coarse_w", 1.0))
        self.fine_w = float(config.get("fine_w", 1.0))
        self.pos_w = float(config.get("pos_w", 1.0))
        self.neg_w = float(config.get("neg_w", 1.0))
        self.focal_alpha = float(config.get("focal_alpha", 0.25))
        self.focal_gamma = float(config.get("focal_gamma", 2.0))
        self.fine_correct_thr = float(config.get("fine_correct_thr", 1.0))
        self.eps = 1e-6

    # ----- coarse -----
    def _focal_loss(self, conf: torch.Tensor, gt: torch.Tensor,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
        """Sigmoid-focal-loss-like binary focal applied to confidences in (0, 1)."""
        c = conf.clamp(self.eps, 1 - self.eps)
        loss_pos = -self.focal_alpha * (1 - c).pow(self.focal_gamma) * c.log()
        loss_neg = -(1 - self.focal_alpha) * c.pow(self.focal_gamma) * (1 - c).log()
        loss_pos_sel = loss_pos[gt]
        loss_neg_sel = loss_neg[~gt]
        if weight is not None:
            w_pos = weight[gt]
            w_neg = weight[~gt]
            loss_pos_sel = loss_pos_sel * w_pos
            loss_neg_sel = loss_neg_sel * w_neg
        return self.pos_w * loss_pos_sel.mean() + self.neg_w * loss_neg_sel.mean()

    def coarse_loss(self, data: dict) -> torch.Tensor:
        conf = data["conf_matrix"]            # [N, L, S]
        gt = data["conf_matrix_gt"]           # [N, L, S] bool
        weight = None
        if "mask0" in data and "mask1" in data:
            # broadcast to [N, L, S] -- mask is True where the token is valid.
            weight = (data["mask0"].flatten(-2)[..., None]
                      * data["mask1"].flatten(-2)[:, None]).float()
        return self._focal_loss(conf, gt, weight)

    # ----- fine -----
    def fine_loss(self, data: dict) -> torch.Tensor:
        pred = data["expec_f"]                # [M, 3] -- (dx, dy, std)
        gt = data.get("expec_f_gt", None)     # [M, 2]
        if gt is None or pred.numel() == 0 or gt.numel() == 0:
            return pred.new_zeros((), requires_grad=True) + 0.0

        # 1. validity: drop matches whose GT residual is outside the fine window.
        correct = gt.abs().max(dim=-1).values < self.fine_correct_thr
        if correct.sum() == 0:
            return pred.new_zeros((), requires_grad=True) + 0.0

        pred_xy = pred[..., :2]
        std = pred[..., 2].clamp(min=self.eps)

        # weighted L2 (variance-weighted, as in LoFTR)
        offset = (pred_xy[correct] - gt[correct])
        loss = ((offset ** 2).sum(-1) / (2 * std[correct] ** 2)).mean()
        return loss

    # ----- combined -----
    def forward(self, data: dict) -> tuple[torch.Tensor, dict]:
        l_c = self.coarse_loss(data)
        l_f = self.fine_loss(data)
        loss = self.coarse_w * l_c + self.fine_w * l_f
        scalars = {"loss": loss.detach(), "loss_c": l_c.detach(), "loss_f": l_f.detach()}
        return loss, scalars
