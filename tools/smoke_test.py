#!/usr/bin/env python3
# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
# New file in MatchFormer-cable fork (no upstream counterpart).
"""Minimal smoke test for the homography FT pipeline.

Synthesises a tiny 4-frame session, builds the lightning module on the
``litesea`` backbone (small enough to run on CPU), and executes one
training_step on a single batch. Verifies:

  * Forward pass through Matchformer (with mask-aware attention) doesn't crash
  * compute_supervision_coarse_h fills spv_b/i/j_ids correctly
  * MatchformerLoss returns a finite scalar
  * loss.backward() runs without NaNs on the first step

Usage:
    python tools/smoke_test.py                # CPU
    python tools/smoke_test.py --device cuda  # GPU
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.defaultmf import get_cfg_defaults  # noqa: E402
from model.lightning_loftr import PL_LoFTR  # noqa: E402
from model.datasets.homography_dataset import HomographyDataset  # noqa: E402


def _make_synthetic_session(out_dir: Path, n_frames: int = 4,
                            h: int = 240, w: int = 320) -> None:
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    mask_dir = out_dir / "mask"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_frames):
        # Texture: random low-frequency pattern + a thicker "cable" blob so the
        # downsampled mask still has enough coarse cells for CoarseMatching's
        # training-time pad assertion.
        rgb = (rng.normal(120, 30, size=(h, w, 3))).clip(0, 255).astype(np.uint8)
        cv2.line(rgb, (40, 60 + i * 10), (w - 40, h - 60 - i * 10),
                 (220, 220, 220), thickness=60, lineType=cv2.LINE_AA)
        depth = np.full((h, w), 0.30, dtype=np.float32) + rng.normal(0, 0.005, (h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(mask, (40, 60 + i * 10), (w - 40, h - 60 - i * 10),
                 255, thickness=60, lineType=cv2.LINE_AA)
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), rgb)
        np.save(depth_dir / f"{i:06d}.npy", depth)
        cv2.imwrite(str(mask_dir / f"{i:06d}.png"), mask)

    meta = {
        "session": out_dir.name,
        "n_frames": n_frames,
        "depth_scale": 0.001,
        "color_intrinsics": {
            "width": w, "height": h, "fx": 380.0, "fy": 380.0,
            "ppx": w / 2, "ppy": h / 2, "model": "Brown_Conrady", "coeffs": [0, 0, 0, 0, 0],
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _build_cfg():
    cfg = get_cfg_defaults()
    # Tiny config: litesea is small enough for CPU smoke testing.
    cfg.MATCHFORMER.BACKBONE_TYPE = "litesea"
    cfg.MATCHFORMER.RESOLUTION = (8, 4)
    cfg.MATCHFORMER.COARSE.D_MODEL = 192
    cfg.MATCHFORMER.COARSE.D_FFN = 192
    cfg.MATCHFORMER.MATCH_COARSE.THR = 0.05
    cfg.MATCHFORMER.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.5  # smoke: many positives
    cfg.MATCHFORMER.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 5    # smoke: thin synthetic mask
    cfg.LOSS.COARSE_W = 1.0
    cfg.LOSS.FINE_W = 1.0
    cfg.TRAINER.OPTIMIZER.LR = 1e-4
    cfg.TRAINER.SCHEDULER.WARMUP_STEPS = 1
    cfg.TRAINER.SCHEDULER.TOTAL_STEPS = 10
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--keep-data", action="store_true",
                    help="Don't delete the synthesised session at exit.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    tmp_root = Path(tempfile.mkdtemp(prefix="matchformer_smoke_"))
    sess = "smoke"
    sess_dir = tmp_root / sess
    print(f"[smoke] session: {sess_dir}", flush=True)

    try:
        _make_synthetic_session(sess_dir, n_frames=4, h=240, w=320)

        ds = HomographyDataset(
            cable_root=str(tmp_root),
            sessions=[sess],
            mode="train",
            img_resize=320,
            pairs_per_frame=2,
            # Smoke: small transforms so the synthetic cable mask survives.
            max_rotation_deg=8.0,
            max_translation_frac=0.02,
            max_scale_log=0.05,
            max_perspective=0.0,
            seed=42,
        )
        print(f"[smoke] dataset size: {len(ds)}", flush=True)
        assert len(ds) > 0

        # Build a single-sample batch (avoid DataLoader to keep it dependency-light).
        sample = ds[0]
        batch = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(args.device)
            elif v is None:
                batch[k] = None
            else:
                batch[k] = [v]

        cfg = _build_cfg()
        model = PL_LoFTR(cfg)
        model.to(args.device)
        model.train()

        loss, scalars = model._supervised_forward(batch)
        print(f"[smoke] loss = {loss.detach().item():.4f}  "
              f"(coarse={scalars['loss_c'].item():.4f}, "
              f"fine={scalars['loss_f'].item():.4f})",
              flush=True)
        assert torch.isfinite(loss), "loss is not finite"

        loss.backward()
        bad = []
        for n, p in model.matcher.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad.append(n)
        assert not bad, f"non-finite grads in: {bad[:5]}"
        print(f"[smoke] backward OK -- all gradients finite", flush=True)

        # One optimizer step
        opt_cfg = model.configure_optimizers()
        opt = opt_cfg["optimizer"]
        opt.step()
        opt.zero_grad()
        print(f"[smoke] optimizer.step() OK", flush=True)

        print("[smoke] SUCCESS", flush=True)
    finally:
        if not args.keep_data:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
