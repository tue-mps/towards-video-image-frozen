"""Streaming inference on NuScenes-style camera pose estimation.

DINOv3 (frozen) + GatedMambaMix recurrent core + per-frame pose head.
At each frame the head predicts a 9D pose delta (3D translation + 6D
rotation as in Zhou et al. CVPR'19). Only a streaming variant exists in
the paper (no offline counterpart).

Output: (1, T, 9) — per-frame pose delta.
  [:, :, :3] = (dx, dy, dz) translation
  [:, :, 3:] = 6D rotation (first two columns of rotation matrix)

The 6D rotation can be turned into a 3x3 matrix via Gram-Schmidt; the
training code does this inside the loss. For this demo we just dump the
raw 9D vector per frame.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.rvm_modules.rvm_wrapper import RVMStreamingCameraPoseWrapper
from scripts_inference.utils import (
    ARCHITECTURES,
    normalization_for,
    build_architecture,
    load_checkpoint,
    load_original_frames,
    load_video_frames,
)
from scripts_inference.viz_utils import (
    draw_compass_arrow,
    draw_text_outlined,
    text_size,
    viz_output_path,
    write_mp4,
)


# ============================================================================
# Configuration
# ============================================================================

# --- Backbone size (use --arch to switch architecture) ---
SIZE = "large"

# --- Video preprocessing ---
NUM_FRAMES   = 16
TARGET_FPS   = 12
FRAME_SIZE   = 224
PATCH_SIZE   = 16
NUM_PATCHES  = (FRAME_SIZE // PATCH_SIZE) ** 2

# --- Readout ---
READOUT_D_MODEL         = 1024
POSE_MLP_HIDDEN_DIM     = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tuned-ckpt", required=True)
    p.add_argument("--arch", choices=ARCHITECTURES, default="dinov3_gatedmambamix",
                   help="Model architecture (see ARCHITECTURES).")
    p.add_argument("--backbone-weights", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated mp4 next to the input video with the per-frame predicted "
                        "translation deltas (dx, dy, dz) in meters drawn on top, plus the translation "
                        "error vs the GT sidecar <video>.pose.json in centimeters if present.")
    return p.parse_args()


def build_model(arch: str, backbone_weights: str = None) -> RVMStreamingCameraPoseWrapper:
    encoder, sequential_core = build_architecture(
        arch, SIZE, NUM_PATCHES, backbone_weights=backbone_weights,
    )
    return RVMStreamingCameraPoseWrapper(
        encoder=encoder,
        d_enc=encoder.output_dim,
        sequential_core=sequential_core,
        d_seq_out=encoder.output_dim,
        freeze_encoder=True,
        readout_d_model=READOUT_D_MODEL,
        pose_mlp_hidden_dim=POSE_MLP_HIDDEN_DIM,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading video: {args.video}")
    mean, std = normalization_for(args.arch)
    n_in = None if args.visualize else NUM_FRAMES
    frames_norm = load_video_frames(args.video, n_in, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
    frames = frames_norm.unsqueeze(0).to(device)
    print(f"  Frames: {tuple(frames.shape)}")
    frames_orig = load_original_frames(args.video, n_in, TARGET_FPS) if args.visualize else None

    print("Building model...")
    model = build_model(args.arch, args.backbone_weights).to(device)
    load_checkpoint(model, args.tuned_ckpt)
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        pose_deltas = model(frames)            # (1, T, 9)

    pose_deltas = pose_deltas[0].cpu().tolist()
    print(f"\nPer-frame pose deltas (streaming, {NUM_FRAMES} frames). Cols: dx dy dz r1 r2 r3 r4 r5 r6")
    for t, p in enumerate(pose_deltas):
        print(f"  t={t:2d}: [{', '.join(f'{v:+.3f}' for v in p)}]")

    if args.visualize:
        save_visualization(args, frames_orig, pose_deltas)


def _load_gt_pose(video_path: str):
    """Load per-frame GT translation deltas (and dt) from <video>.pose.json
    if present. Returns (deltas, dt_s_or_None) or None when no sidecar exists.

    `dt_s[t]` is the wall-clock gap (seconds) between video frame t-1 and t,
    which can be irregular when the source dataset mixes sample rates
    (NuScenes keyframes at 2 Hz + sweeps at ~12 Hz).
    """
    import json
    sidecar = os.path.splitext(video_path)[0] + ".pose.json"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        spec = json.load(f)
    gt = spec.get("translation_deltas_m")
    if gt is None:
        return None
    return gt, spec.get("dt_s")


def save_visualization(args, frames_orig, pose_deltas):
    """Save an annotated mp4 showing per-frame translation predictions vs GT.

    For every frame draws a compass-style arrow indicating the predicted
    horizontal-plane motion direction, followed by the predicted distance
    and the GT err in cm (if a sidecar ``<video>.pose.json`` is present).

    `frames_orig` is the original-resolution uint8 RGB (T, H, W, 3) ndarray.
    """
    T = len(pose_deltas)

    sidecar = _load_gt_pose(args.video)
    if sidecar is None:
        gt = None
        dt_s = None
        print(f"\n[visualize] no <video>.pose.json next to {args.video}; "
              f"skipping err-vs-GT, using TARGET_FPS for velocity.")
    else:
        gt, dt_s = sidecar
        if len(gt) < T:
            print(f"[visualize] WARNING: GT has {len(gt)} entries but video has {T}; "
                  f"truncating to {len(gt)}.")
            T = len(gt)
        if dt_s is not None:
            print(f"[visualize] using per-frame dt_s from sidecar "
                  f"(min={min(d for d in dt_s if d > 0):.3f}s, "
                  f"max={max(dt_s):.3f}s) for velocity.")
        else:
            print(f"[visualize] sidecar has no dt_s; falling back to TARGET_FPS for velocity.")

    out_frames = []
    for t in range(T):
        frame = frames_orig[t].copy()
        H, W = frame.shape[:2]
        font_scale = max(0.45, W / 700.0)
        thickness = max(1, int(font_scale * 1.5))
        # Slightly thicker outline that scales with text size.
        outline = thickness * 2 + 1
        _, h, _ = text_size("Ag", font_scale=font_scale, thickness=thickness)

        dx, _, dz = pose_deltas[t][:3]
        dist_m = (dx * dx + dz * dz) ** 0.5
        # Velocity = distance / time. Prefer the sidecar's per-frame `dt_s`
        # (wall-clock gap from the source dataset) since NuScenes mixes
        # 2 Hz keyframes with ~12 Hz sweeps; if it isn't available, fall
        # back to a uniform 1 / TARGET_FPS gap.
        if dt_s is not None and t < len(dt_s) and dt_s[t] > 0:
            vel_kmh = (dist_m / dt_s[t]) * 3.6
        else:
            vel_kmh = dist_m * TARGET_FPS * 3.6
        # The compass is a steering indicator: up = going straight, tilted left
        # = turning left, tilted right = turning right, dot = stopped.
        # For a car driving forward the per-frame translation is dominated by
        # dz (~1 m / frame at 12 fps); the lateral component dx is small
        # relative to dz (a 1 deg/frame yaw rate -> only ~1.7 cm lateral).
        # To make turns legible we use the per-frame yaw angle directly
        # (atan2(dx, dz)) and amplify it for visualization, clamped so the
        # arrow never tips past 60 degrees from straight up.
        STOPPED_THRESHOLD_M = 0.02       # under 2 cm / frame -> dot
        YAW_AMP = 6.0                    # 1 deg actual -> 6 deg visual
        YAW_CLAMP = math.radians(60.0)
        # t = 0 has no preceding frame, so the model's translation output
        # there is unconstrained. Show nothing for dist/err/vel and draw a
        # neutral dot on the compass.
        no_prev_frame = (t == 0)
        if no_prev_frame or dist_m < STOPPED_THRESHOLD_M:
            direction = (0.0, 0.0)        # dot
        else:
            yaw = math.atan2(dx, dz)      # 0 = straight, +ve = right
            yaw_vis = max(-YAW_CLAMP, min(YAW_CLAMP, yaw * YAW_AMP))
            direction = (math.sin(yaw_vis), -math.cos(yaw_vis))
        radius = max(14, int(0.05 * min(H, W)))
        compass_x = 8 + radius
        compass_y = 8 + radius
        draw_compass_arrow(frame, (compass_x, compass_y), direction, radius=radius,
                           color=(255, 255, 255), thickness=max(1, thickness))

        white = (255, 255, 255)
        # Predictions in green, errors in a slightly desaturated orange.
        PRED_COLOR = (100, 230, 120)
        ERR_COLOR = (255, 165, 70)

        if no_prev_frame:
            err_piece = ("err -", ERR_COLOR) if gt is not None else None
            dist_piece = ("dist -", PRED_COLOR)
            vel_piece = ("vel: -", PRED_COLOR)
        else:
            dist_piece = (f"dist {dist_m:.2f} m", PRED_COLOR)
            vel_piece = (f"vel: {vel_kmh:.1f} km/h", PRED_COLOR)
            if gt is not None:
                gx, _, gz = gt[t][:3]
                err_cm = ((dx - gx) ** 2 + (pose_deltas[t][1] - gt[t][1]) ** 2 + (dz - gz) ** 2) ** 0.5 * 100.0
                err_piece = (f"err {err_cm:.1f} cm", ERR_COLOR)
            else:
                err_piece = None

        pieces = [dist_piece]
        if err_piece is not None:
            pieces += [(" | ", white), err_piece]
        pieces += [(" | ", white), vel_piece]

        # Auto-shrink so the whole line fits within the frame width.
        text_start_x = compass_x + radius + 6
        margin_right = 6
        max_text_w = W - text_start_x - margin_right
        for _ in range(8):
            total_w = sum(text_size(t, font_scale=font_scale, thickness=thickness)[0]
                          for t, _ in pieces)
            if total_w <= max_text_w or font_scale <= 0.4:
                break
            font_scale *= 0.9
            thickness = max(1, int(font_scale * 1.5))
            outline = thickness * 2 + 1
        _, h, _ = text_size("Ag", font_scale=font_scale, thickness=thickness)

        # Place text just to the right of the compass, vertically centered.
        text_y = compass_y + h // 2
        x = text_start_x
        for text, color in pieces:
            draw_text_outlined(frame, text, (x, text_y),
                               fg=color, font_scale=font_scale,
                               thickness=thickness, outline_thickness=outline)
            w, _, _ = text_size(text, font_scale=font_scale, thickness=thickness)
            x += w
        out_frames.append(frame)

    out_path = viz_output_path(args.video, args.arch)
    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({len(out_frames)} frames @ {TARGET_FPS} fps)")


if __name__ == "__main__":
    main()
