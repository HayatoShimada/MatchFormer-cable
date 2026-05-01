#!/usr/bin/env python3
# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see LICENSE in this directory.
# New file in MatchFormer-cable fork (no upstream counterpart).
"""MatchFormer-cable fine-tuning entrypoint.

Example:

    python train.py \\
        --data-cfg config/data/cable_default.py \\
        --pretrained /opt/matchformer-weights/indoor-large-SEA.ckpt \\
        --train-sessions 20260501_141200,20260502_092030 \\
        --val-sessions   20260503_080000 \\
        --max-epochs 20 \\
        --batch-size 4 \\
        --gpus 1 \\
        --output-dir /app/datasets/matchformer_ft/runs/v1
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.defaultmf import get_cfg_defaults  # noqa: E402
from model.lightning_loftr import PL_LoFTR  # noqa: E402
from model.datasets.cable_sequence import CableSequenceDataset  # noqa: E402


def _load_data_cfg(path: Path):
    spec = importlib.util.spec_from_file_location("data_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.cfg


def _collate(batch):
    """Stack tensors; keep masks as bool tensors. Drop incompatible items.

    The default collate stacks per-key. Masks may be None when use_mask=False;
    we stack them only when present in every sample.
    """
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[k] = torch.stack(vals, dim=0)
        elif vals[0] is None:
            out[k] = None
        else:
            out[k] = vals  # list of strings / tuples
    return out


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-cfg", type=Path, required=True)
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--train-sessions", type=str, default="",
                   help="Comma-separated session names under DATASET.CABLE_ROOT.")
    p.add_argument("--val-sessions", type=str, default="")
    p.add_argument("--cable-root", type=str, default="",
                   help="Override DATASET.CABLE_ROOT.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--output-dir", type=Path,
                   default=Path("/app/datasets/matchformer_ft/runs/latest"))
    p.add_argument("--precision", type=int, default=32, choices=(16, 32))
    p.add_argument("--check-val-every-n-epoch", type=int, default=1)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _load_data_cfg(args.data_cfg)
    if args.cable_root:
        cfg.DATASET.CABLE_ROOT = args.cable_root
    if args.train_sessions:
        cfg.DATASET.CABLE_SESSIONS_TRAIN = [s for s in args.train_sessions.split(",") if s]
    if args.val_sessions:
        cfg.DATASET.CABLE_SESSIONS_VAL = [s for s in args.val_sessions.split(",") if s]

    cfg.TRAINER.SCHEDULER.TOTAL_STEPS = (
        max(1, args.max_epochs)
        * max(1, cfg.DATASET.CABLE_PAIRS_PER_SESSION
                * max(1, len(cfg.DATASET.CABLE_SESSIONS_TRAIN)))
        // max(1, args.batch_size * max(1, args.gpus))
    )

    pl.seed_everything(cfg.TRAINER.SEED)

    if not cfg.DATASET.CABLE_SESSIONS_TRAIN:
        raise SystemExit(
            "No training sessions configured. "
            "Pass --train-sessions A,B or set DATASET.CABLE_SESSIONS_TRAIN.")

    train_ds = CableSequenceDataset(
        cable_root=cfg.DATASET.CABLE_ROOT,
        sessions=cfg.DATASET.CABLE_SESSIONS_TRAIN,
        mode="train",
        img_resize=cfg.DATASET.CABLE_IMG_RESIZE,
        pair_stride=(cfg.DATASET.CABLE_PAIR_STRIDE_MIN,
                     cfg.DATASET.CABLE_PAIR_STRIDE_MAX),
        pairs_per_session=cfg.DATASET.CABLE_PAIRS_PER_SESSION,
        use_mask=cfg.DATASET.CABLE_USE_MASK,
        require_pose=cfg.DATASET.CABLE_REQUIRE_POSE,
        seed=cfg.TRAINER.SEED,
    )
    val_ds = (CableSequenceDataset(
        cable_root=cfg.DATASET.CABLE_ROOT,
        sessions=cfg.DATASET.CABLE_SESSIONS_VAL,
        mode="val",
        img_resize=cfg.DATASET.CABLE_IMG_RESIZE,
        pair_stride=(cfg.DATASET.CABLE_PAIR_STRIDE_MIN,
                     cfg.DATASET.CABLE_PAIR_STRIDE_MAX),
        pairs_per_session=max(20, cfg.DATASET.CABLE_PAIRS_PER_SESSION // 5),
        use_mask=cfg.DATASET.CABLE_USE_MASK,
        require_pose=cfg.DATASET.CABLE_REQUIRE_POSE,
        seed=cfg.TRAINER.SEED + 1,
    ) if cfg.DATASET.CABLE_SESSIONS_VAL else None)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = (DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=False,
    ) if val_ds is not None else None)

    model = PL_LoFTR(cfg, pretrained_ckpt=args.pretrained)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="cable-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3, monitor="val/loss" if val_loader else "train/loss",
        mode="min", save_last=True,
    )
    lr_cb = pl.callbacks.LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus if torch.cuda.is_available() else 0,
        precision=args.precision,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(args.output_dir),
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
