# MatchFormer-cable: forked for cable-specific fine-tuning

This is a hard fork of [InSAI-Lab/MatchFormer](https://github.com/InSAI-Lab/MatchFormer)
at upstream commit **`1b2da5c8ecb0c1fc918f8d5d64fcce1beeab2ba2`** ("update").

## License

Upstream MatchFormer is licensed under the **Apache License, Version 2.0**.
This fork remains under the same license; the upstream `LICENSE` file is
preserved verbatim and a `NOTICE` file documents attribution to upstream
MatchFormer and to LoFTR (whose training-time supervision and loss
formulations we re-implement under their own Apache-2.0 licence).

Per Apache-2.0 §4(b), every source file modified relative to upstream
carries a notice at the top of the file in the form:

```
# Modifications copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see LICENSE in this directory.
# Original file: <upstream path>  (commit 1b2da5c)
```

New files added by the fork carry a standalone Apache-2.0 header.

## Why a fork

The upstream repository ships **inference / test code only**: `test.py`,
`PL_LoFTR.test_step`, no `training_step`, no supervision, no loss, no
training-time data module hooks. It also assumes both images are valid
end-to-end and bakes that into the backbone's full-image attention.

Our use case is sub-mm cable localisation against a *segmented* template,
where the query image's background is full of distractors that the model has
never seen during pretraining. The differences from upstream are deep enough
to be a fork rather than an overlay:

1. **Mask-aware attention.** The backbone's self/cross-attention is gated by
   binary masks so non-cable tokens cannot leak into matched features.
   (Upstream only masks the *coarse correlation* matrix.)
2. **Asymmetric masks.** `mask0` (template) can be supplied alone; `mask1`
   (query) defaults to all-ones. Upstream requires both or neither.
3. **Training pipeline.** LoFTR-style coarse + fine GT supervision computed
   from depth and relative pose, focal coarse loss + L2 fine loss, AdamW +
   warmup + cosine LR, all inside an extended Lightning module.
4. **Custom Dataset.** Reads RGB+depth+mask sequences captured by
   `recorder.py` (no MegaDepth/ScanNet dependency).

## Layout vs. upstream

```
MatchFormer-cable/
├── FORK.md                                  ← this file
├── README.md                                ← upstream README, kept verbatim
├── config/
│   ├── defaultmf.py                         (modified: + LOSS / + TRAINER.OPTIMIZER)
│   ├── data/                                (upstream, unchanged)
│   └── train_cable.py                       (NEW: training defaults for cable FT)
├── model/
│   ├── matchformer.py                       (modified: asymmetric masks, mask passthrough)
│   ├── backbone/                            (modified: mask-aware attention)
│   ├── lightning_loftr.py                   (modified: + training_step + configure_optimizers)
│   ├── datasets/
│   │   ├── megadepth.py / scannet.py        (upstream, unchanged)
│   │   └── cable_sequence.py                (NEW: recorder-output Dataset)
│   ├── losses/
│   │   └── matchformer_loss.py              (NEW: focal coarse + L2 fine)
│   ├── supervision.py                       (NEW: depth + pose -> coarse/fine GT)
│   └── utils/                               (upstream, unchanged)
├── recorder.py                              (NEW: D405 video recorder w/ DINOv2+SAM2 mask)
├── train.py                                 (NEW: training entrypoint)
└── test.py                                  (upstream, unchanged)
```

`weight/` is intentionally not committed; the `indoor-large-SEA.ckpt`
symlink under `/opt/matchformer-weights/` already points at the upstream
checkpoint and is loaded as the FT initialisation.

## Re-syncing with upstream

If upstream publishes a new commit:

```bash
cd /tmp && git clone https://github.com/InSAI-Lab/MatchFormer.git mf-upstream
diff -ru mf-upstream /app/third_party/MatchFormer-cable | less
```

Manually merge bug fixes; do not blanket-overwrite — the modified files above
are deeply diverged.
