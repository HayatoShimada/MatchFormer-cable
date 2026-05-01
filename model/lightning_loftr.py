# Modifications copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see ../LICENSE.
# Original file: model/lightning_loftr.py (upstream commit 1b2da5c).
#
# Changes from upstream:
#   * Added training_step, validation_step and configure_optimizers so the
#     module can be used with `pl.Trainer.fit(...)` instead of test-only.
#   * Wires LoFTR-style coarse + fine GT supervision (model.supervision) and
#     the focal+L2 loss (model.losses.MatchformerLoss).
#   * Loads checkpoints whose state-dict was saved as either a raw matcher
#     dict or a Lightning checkpoint with `state_dict` and `matcher.` prefix.
#   * AdamW + linear-warmup + cosine LR schedule, configurable via
#     config.TRAINER.OPTIMIZER and config.TRAINER.SCHEDULER.
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl

from .matchformer import Matchformer
from .losses import MatchformerLoss
from .supervision import compute_supervision_coarse, compute_supervision_fine
from .supervision_homography import (
    compute_supervision_coarse_h, compute_supervision_fine_h,
)
from .utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from .utils.comm import gather
from .utils.misc import lower_config, flattenList
from .utils.profiler import PassThroughProfiler


def _load_pretrained(ckpt_path: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k.replace("matcher.", ""): v for k, v in state.items()}


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        self.config = config
        _config = lower_config(self.config)
        self.profiler = profiler or PassThroughProfiler()

        self.matcher = Matchformer(config=_config["matchformer"])

        # Loss (training only — instantiated unconditionally; harmless at test).
        loss_cfg = _config.get("loss", {}) if isinstance(_config, dict) else {}
        self.loss_fn = MatchformerLoss(loss_cfg)

        if pretrained_ckpt:
            sd = _load_pretrained(pretrained_ckpt)
            missing, unexpected = self.matcher.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"missing keys ({len(missing)}): {missing[:3]} ...")
            if unexpected:
                logger.warning(f"unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        self.dump_dir = dump_dir

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def _supervised_forward(self, batch: dict) -> torch.Tensor:
        """Run forward + supervision + loss. Mutates `batch`.

        Picks the supervision flavour from ``batch['data_mode']`` (a list of
        strings produced by the Dataset). Falls back to depth+pose mode for
        legacy callers that didn't set the key.
        """
        scale_c = int(self.config.MATCHFORMER.RESOLUTION[0])
        scale_f = int(self.config.MATCHFORMER.RESOLUTION[1])
        fine_w = int(self.config.MATCHFORMER.FINE_WINDOW_SIZE)

        mode = batch.get("data_mode", "pose")
        if isinstance(mode, list):
            mode = mode[0] if mode else "pose"

        if mode == "homography":
            compute_supervision_coarse_h(batch, scale_c=scale_c)
            self.matcher(batch)
            compute_supervision_fine_h(batch, fine_window_size=fine_w, scale_f=scale_f)
        else:
            compute_supervision_coarse(batch, scale_c=scale_c)
            self.matcher(batch)
            compute_supervision_fine(batch, fine_window_size=fine_w, scale_f=scale_f)

        loss, scalars = self.loss_fn(batch)
        return loss, scalars

    def training_step(self, batch, batch_idx):
        loss, scalars = self._supervised_forward(batch)
        for k, v in scalars.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=(k == "loss"))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, scalars = self._supervised_forward(batch)
        for k, v in scalars.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        return scalars

    # ------------------------------------------------------------------ #
    #  Optimizer + scheduler
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        opt_cfg = self.config.TRAINER.OPTIMIZER
        sch_cfg = self.config.TRAINER.SCHEDULER

        params = [p for p in self.matcher.parameters() if p.requires_grad]
        if opt_cfg.NAME.lower() == "adamw":
            optim = torch.optim.AdamW(params, lr=opt_cfg.LR,
                                      weight_decay=opt_cfg.WEIGHT_DECAY)
        elif opt_cfg.NAME.lower() == "adam":
            optim = torch.optim.Adam(params, lr=opt_cfg.LR,
                                     weight_decay=opt_cfg.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_cfg.NAME}")

        # Linear warmup -> cosine decay over total steps.
        warmup_steps = int(sch_cfg.WARMUP_STEPS)
        total_steps = int(sch_cfg.TOTAL_STEPS)
        min_lr_ratio = float(sch_cfg.MIN_LR_RATIO)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            if total_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cos = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ------------------------------------------------------------------ #
    #  Test (kept verbatim from upstream behaviour)
    # ------------------------------------------------------------------ #
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)
            compute_pose_errors(batch, self.config)

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])
            dumps = flattenList(gather(_dumps))
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
