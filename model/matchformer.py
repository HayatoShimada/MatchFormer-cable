# Modifications copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
# Original file: model/matchformer.py (upstream commit 1b2da5c).
#
# Changes from upstream:
#   * Accept asymmetric image-resolution masks via data['mask0_full'] /
#     data['mask1_full']. Either side may be None (in which case the missing
#     side defaults to all-True), enabling "segmented template vs. unsegmented
#     query" matching.
#   * Pass image-resolution masks to the backbone so its self/cross-attention
#     can drop background tokens before the coarse correlation. Backbones that
#     do not yet support a `mask` kwarg are silently called without it.
#   * Auto-derive coarse-resolution mask0/mask1 from mask0_full/mask1_full
#     when not supplied, so the existing CoarseMatching softmax gating keeps
#     working without callers having to compute the downsampling.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .backbone.fine_preprocess import FinePreprocess
from .backbone.coarse_matching import CoarseMatching
from .backbone.fine_matching import FineMatching
from einops.einops import rearrange


def _ones_mask_like(image: torch.Tensor) -> torch.Tensor:
    return torch.ones(image.size(0), image.size(2), image.size(3),
                      dtype=torch.bool, device=image.device)


def _coarse_mask_from_full(mask_full: torch.Tensor, h_c: int, w_c: int) -> torch.Tensor:
    """Downsample a binary [B, H, W] mask to coarse [B, H_c, W_c] via max-pool.
    A coarse cell is "valid" if any of its sub-pixels were valid.
    """
    m = mask_full.float().unsqueeze(1)
    m = F.adaptive_max_pool2d(m, output_size=(h_c, w_c))
    return m.squeeze(1) > 0.5


class Matchformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching()

    def _resolve_full_masks(self, data):
        m0 = data.get("mask0_full", None)
        m1 = data.get("mask1_full", None)
        if m0 is None and m1 is None:
            return None, None
        if m0 is None:
            m0 = _ones_mask_like(data["image0"])
        if m1 is None:
            m1 = _ones_mask_like(data["image1"])
        return m0.to(torch.bool), m1.to(torch.bool)

    def _call_backbone(self, x, mask):
        try:
            return self.backbone(x, mask=mask)
        except TypeError:
            return self.backbone(x)

    def forward(self, data):
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:],
        })

        mask0_full, mask1_full = self._resolve_full_masks(data)

        if data['hw0_i'] == data['hw1_i']:
            backbone_input = torch.cat([data['image0'], data['image1']], dim=0)
            backbone_mask = (None if mask0_full is None
                             else torch.cat([mask0_full, mask1_full], dim=0))
            feats_c, feats_f = self._call_backbone(backbone_input, backbone_mask)
            feat_c0, feat_c1 = feats_c.split(data['bs'])
            feat_f0, feat_f1 = feats_f.split(data['bs'])
        else:
            feat_c0, feat_f0 = self._call_backbone(data['image0'], mask0_full)
            feat_c1, feat_f1 = self._call_backbone(data['image1'], mask1_full)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
        })

        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        # Coarse-resolution masks. Prefer caller-supplied data['mask0/1'],
        # otherwise derive from the full-resolution masks.
        mask_c0 = mask_c1 = None
        if 'mask0' in data and 'mask1' in data:
            mask_c0 = data['mask0'].flatten(-2)
            mask_c1 = data['mask1'].flatten(-2)
        elif mask0_full is not None:
            h0c, w0c = data['hw0_c']
            h1c, w1c = data['hw1_c']
            data['mask0'] = _coarse_mask_from_full(mask0_full, h0c, w0c)
            data['mask1'] = _coarse_mask_from_full(mask1_full, h1c, w1c)
            mask_c0 = data['mask0'].flatten(-2)
            mask_c1 = data['mask1'].flatten(-2)

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data)

        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
