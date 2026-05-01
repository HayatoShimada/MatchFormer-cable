# Copyright 2026 Hayato Shimada.
# Licensed under the Apache License, Version 2.0; see LICENSE in this directory.
# New file in MatchFormer-cable fork (no upstream counterpart).
"""Video-style recorder for MatchFormer fine-tuning datasets.

Records a synchronized RGB + depth + cable-mask sequence from the D405 using
the same DINOv2 + SAM2 segmentation pipeline as ``perception/seg/realtime_scene.py``.
A configurable start delay lets the operator finish positioning the camera
before recording begins.

Output layout (one directory per session):

    <out-root>/<session>/
        rgb/000000.png        BGR (uint8)
        depth/000000.npy      depth in metres (float32, HxW)
        mask/000000.png       cable mask (uint8 0/255)
        meta.json             intrinsics, depth scale, fps, timestamps,
                              CLI args, device info

The mask is the union of all DINOv2+SAM2 detections for that frame. Frames
where no cable was detected are still saved (with an all-zero mask) so the
sequence stays continuous; you can drop them later in dataset.py if needed.

Camera pose is *not* recorded here — for LoFTR-style supervised fine-tuning
you need pose per frame. Either:
  * mount the D405 on the UR and join FK + hand-eye into ``poses.csv``
    after recording, or
  * use self-supervised homography pairs (see docs/MATCHFORMER_FT.md).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Reuse seg pipeline and RealSense helpers from the host repo (/app).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from perception.camera.realsense_capture import (  # noqa: E402
    build_runtime,
    enumerate_devices,
    get_aligned_frame_bundle,
    select_serials,
)
from perception.seg.realtime_scene import load_model, run_inference  # noqa: E402


DEFAULT_OUT_ROOT = Path("/app/datasets/matchformer_ft")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a segmented RGB+depth video for MatchFormer fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- camera ----
    p.add_argument("--serial", default=os.environ.get("REALSENSE_SERIAL", ""),
                   help="RealSense serial (empty = first device).")
    p.add_argument("--width", type=int, default=int(os.environ.get("REALSENSE_WIDTH", "640")))
    p.add_argument("--height", type=int, default=int(os.environ.get("REALSENSE_HEIGHT", "480")))
    p.add_argument("--fps", type=int, default=int(os.environ.get("REALSENSE_FPS", "30")),
                   help="Streaming FPS for the camera (recording rate is --record-fps).")
    p.add_argument("--rs-settings", type=str,
                   default=os.environ.get("REALSENSE_SETTINGS",
                                          "/app/perception/seg/realsense_settings.json"))
    p.add_argument("--min-depth", type=float, default=0.02)
    p.add_argument("--max-depth", type=float, default=0.50)
    p.add_argument("--warmup-frames", type=int, default=5)

    # ---- segmentation (DINOv2 + SAM2) — same flags as realtime_scene.py ----
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--ref-image", required=True, type=Path,
                   help="Reference cable image (RGB) for the DINOv2 prototype.")
    p.add_argument("--ref-mask", required=True, type=Path,
                   help="Reference cable mask (uint8 0/255) for the DINOv2 prototype.")
    p.add_argument("--similarity-thresh", type=float, default=0.55)
    p.add_argument("--max-det", type=int, default=10)
    p.add_argument("--cross-class-iou", type=float, default=0.0)
    p.add_argument("--sam2-checkpoint", type=str,
                   default="/opt/sam2-weights/sam2.1_hiera_large.pt")
    p.add_argument("--sam2-config", type=str,
                   default="configs/sam2.1/sam2.1_hiera_l.yaml")

    # ---- recording schedule ----
    p.add_argument("--start-delay", type=float, default=10.0,
                   help="Wait this many seconds *after the camera and models are ready* "
                        "before the first frame is saved.")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Total recording length in seconds.")
    p.add_argument("--record-fps", type=float, default=10.0,
                   help="Target rate at which frames are *saved* (camera streams faster).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Hard cap on saved frames (0 = unlimited within --duration).")

    # ---- output ----
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help="Parent directory for sessions.")
    p.add_argument("--session", default="",
                   help="Session name (default: timestamp).")
    p.add_argument("--mask-dilate", type=int, default=0,
                   help="Dilate the cable mask by N pixels before saving.")
    p.add_argument("--require-mask", action="store_true",
                   help="Drop frames where the cable mask is empty.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _make_session_dir(args: argparse.Namespace) -> Path:
    name = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_root / name
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    return out


def _union_mask(detections: list[dict[str, Any]], hw: tuple[int, int],
                dilate_px: int = 0) -> np.ndarray:
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        m = det.get("mask")
        if m is None:
            continue
        mask[m.astype(bool)] = 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate_px)
    return mask


def main() -> int:
    args = parse_args()
    out_dir = _make_session_dir(args)
    target_n = (int(args.duration * args.record_fps) if not args.max_frames
                else min(args.max_frames, int(args.duration * args.record_fps)))
    print(
        f"[recorder] output: {out_dir}\n"
        f"[recorder] schedule: start_delay={args.start_delay:.1f}s, "
        f"duration={args.duration:.1f}s, record_fps={args.record_fps:.1f} "
        f"(~{target_n} frames)",
        file=sys.stderr,
    )

    # ---- camera ----
    devices = enumerate_devices()
    if not devices:
        print("[recorder] no RealSense device found", file=sys.stderr)
        return 1
    serial = select_serials(devices, [args.serial] if args.serial else None,
                            expected_count=1)[0]
    info = next(d for d in devices if d.serial_number == serial)
    runtime = build_runtime(
        info, args.width, args.height, args.fps,
        settings_json=(args.rs_settings.strip() or None) if isinstance(args.rs_settings, str) else args.rs_settings,
    )

    # ---- segmentation models ----
    print("[recorder] loading DINOv2 + SAM2...", file=sys.stderr)
    model = load_model(args, debug=args.debug)

    # ---- warmup ----
    for _ in range(args.warmup_frames):
        get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)

    # ---- start-delay countdown (camera and models are warm) ----
    if args.start_delay > 0:
        print(f"[recorder] start-delay {args.start_delay:.1f}s ...", file=sys.stderr)
        t_end = time.monotonic() + args.start_delay
        while time.monotonic() < t_end:
            # keep pulling frames so the auto-exposure does not freeze on idle
            get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)

    # ---- record ----
    shutdown = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: shutdown.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: shutdown.update(flag=True))

    period = 1.0 / max(args.record_fps, 0.1)
    t_start = time.monotonic()
    next_save = t_start
    frame_idx = 0
    timestamps: list[dict[str, float]] = []
    intrinsics_dump: dict[str, Any] | None = None
    depth_scale_dump: float | None = None

    try:
        while not shutdown["flag"]:
            now = time.monotonic()
            if now - t_start >= args.duration:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            bundle = get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)
            if now < next_save:
                continue  # drop frame: ahead of schedule

            color_bgr = bundle["color"]
            depth_u16 = bundle["depth"]
            depth_scale = float(bundle["depth_scale"])
            depth_m = depth_u16.astype(np.float32) * depth_scale
            depth_m[(depth_m < args.min_depth) | (depth_m > args.max_depth)] = 0.0

            # segment cable
            detections = run_inference(model, color_bgr, args)
            mask = _union_mask(detections, color_bgr.shape[:2], dilate_px=args.mask_dilate)
            if args.require_mask and not mask.any():
                if args.debug:
                    print(f"[recorder] frame skipped (no cable detected)", file=sys.stderr)
                continue

            # save
            stem = f"{frame_idx:06d}"
            cv2.imwrite(str(out_dir / "rgb" / f"{stem}.png"), color_bgr)
            np.save(out_dir / "depth" / f"{stem}.npy", depth_m.astype(np.float32))
            cv2.imwrite(str(out_dir / "mask" / f"{stem}.png"), mask)

            timestamps.append({
                "frame": frame_idx,
                "wall_s": now - t_start,
                "camera_ts_ms": float(bundle.get("timestamp_ms", 0.0)),
                "camera_frame": int(bundle.get("frame_number", 0)),
            })
            if intrinsics_dump is None:
                intrinsics_dump = bundle["color_intrinsics"]
                depth_scale_dump = depth_scale

            frame_idx += 1
            next_save += period
            if args.debug and frame_idx % 10 == 0:
                fps = frame_idx / max(1e-3, (now - t_start))
                print(f"[recorder] frame {frame_idx}  {fps:.1f} fps  mask_px={int((mask>0).sum())}",
                      file=sys.stderr)
    finally:
        runtime.pipeline.stop()

    # ---- meta ----
    meta = {
        "session": out_dir.name,
        "duration_s": time.monotonic() - t_start,
        "n_frames": frame_idx,
        "record_fps_target": args.record_fps,
        "stream_fps": args.fps,
        "depth_scale": depth_scale_dump,
        "color_intrinsics": intrinsics_dump,
        "device": {
            "serial": info.serial_number,
            "name": info.name,
            "firmware": info.firmware_version,
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "timestamps": timestamps,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[recorder] wrote {frame_idx} frames to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
