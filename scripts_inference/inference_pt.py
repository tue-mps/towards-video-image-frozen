"""Streaming inference on Perception Test-style point tracking.

DINOv3 (frozen) + GatedMambaMix recurrent core + per-frame point readout.
The readout attends to each frame's spatial tokens independently; temporal
context comes from the recurrent core's hidden state.

The query is supplied as a JSON sidecar:
    {"query_points": [[x1, y1], [x2, y2], ...], "query_frame": 0}
Coordinates are normalized to [0, 1].
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.rvm_modules.rvm_wrapper import RVMStreamingPointTrackingWrapper
from scripts_inference.utils import (
    ARCHITECTURES,
    normalization_for,
    build_architecture,
    load_checkpoint,
    load_original_frames,
    load_video_frames,
    load_query_points,
)
from scripts_inference.viz_utils import (
    PALETTE,
    crop_square_norm_to_orig_norm,
    draw_points_norm,
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
READOUT_D_MODEL              = 1024
READOUT_NUM_HEADS            = 8
READOUT_NUM_LAYERS           = 1
NUM_FOURIER_FREQUENCIES      = 16
MLP_HIDDEN_DIM               = 512
PREDICT_UNCERTAINTY          = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--query", required=True, help="JSON sidecar with 'query_points' and optional 'query_frame'.")
    p.add_argument("--tuned-ckpt", required=True)
    p.add_argument("--arch", choices=ARCHITECTURES, default="dinov3_gatedmambamix",
                   help="Model architecture (see ARCHITECTURES).")
    p.add_argument("--backbone-weights", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated mp4 next to the input video with the per-frame predicted "
                        "query-point positions. Each query gets its own color; points with low "
                        "visibility are drawn as hollow rings.")
    return p.parse_args()


def build_model(arch: str, backbone_weights: str = None) -> RVMStreamingPointTrackingWrapper:
    encoder, sequential_core = build_architecture(
        arch, SIZE, NUM_PATCHES, backbone_weights=backbone_weights,
    )
    return RVMStreamingPointTrackingWrapper(
        encoder=encoder,
        d_enc=encoder.output_dim,
        readout_d_model=READOUT_D_MODEL,
        readout_num_heads=READOUT_NUM_HEADS,
        readout_num_layers=READOUT_NUM_LAYERS,
        num_fourier_frequencies=NUM_FOURIER_FREQUENCIES,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        predict_uncertainty=PREDICT_UNCERTAINTY,
        sequential_core=sequential_core,
        freeze_encoder=True,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading video: {args.video}")
    mean, std = normalization_for(args.arch)
    n_in = None if args.visualize else NUM_FRAMES
    frames_norm = load_video_frames(args.video, n_in, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
    frames = frames_norm.unsqueeze(0).to(device)
    frames_orig = load_original_frames(args.video, n_in, TARGET_FPS) if args.visualize else None
    query_pts, query_frame = load_query_points(args.query)
    query_pts = query_pts.to(device)
    query_frame = query_frame.to(device)
    print(f"  Frames: {tuple(frames.shape)}, query_points: {tuple(query_pts.shape)}, query_frame: {query_frame.tolist()}")

    print("Building model...")
    model = build_model(args.arch, args.backbone_weights).to(device)
    load_checkpoint(model, args.tuned_ckpt)
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        out = model(frames, query_pts, query_frame)

    print(f"\nPer-frame predictions (streaming protocol):")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")

    if args.visualize:
        save_visualization(args, frames_orig, out)


def save_visualization(args, frames_orig, out):
    """Save an annotated mp4 with per-frame predicted point positions.

    Predictions come out of the model in [0, 1] normalized to the
    center-cropped square model input; we map them to [0, 1] of the
    original rectangular frame for drawing.

    Output shapes from the wrapper for a multi-query input:
      pred_points     : (1, Q, T, 2)
      pred_visibility : (1, Q, T)  -- logits; visible if > 0
    """
    pred_points = out["pred_points"]
    pred_vis = out.get("pred_visibility")
    if pred_points.dim() == 4:
        pred_points = pred_points[0]
    elif pred_points.dim() == 3:
        pred_points = pred_points.unsqueeze(0)
    if pred_vis is not None:
        pred_vis = pred_vis[0] if pred_vis.dim() == 3 else pred_vis.unsqueeze(0)

    Q, T, _ = pred_points.shape
    pts_np = pred_points.detach().cpu().numpy()
    vis_np = (pred_vis.detach().cpu().numpy() > 0.0) if pred_vis is not None else None

    out_frames = []
    for t in range(T):
        frame = frames_orig[t].copy()
        H, W = frame.shape[:2]
        points = [
            crop_square_norm_to_orig_norm(float(pts_np[q, t, 0]), float(pts_np[q, t, 1]), H, W)
            for q in range(Q)
        ]
        colors = [PALETTE[q % len(PALETTE)] for q in range(Q)]
        visible = [bool(vis_np[q, t]) for q in range(Q)] if vis_np is not None else None
        draw_points_norm(frame, points, colors, visible=visible, radius=max(4, min(H, W) // 80))
        font_scale = max(0.5, W / 800.0)
        thickness = max(1, int(font_scale * 1.5))
        _, h, _ = text_size(f"t={t}", font_scale=font_scale, thickness=thickness)
        draw_text_outlined(frame, f"t={t}", org=(8, h + 8),
                           fg=(255, 255, 255), font_scale=font_scale, thickness=thickness)
        out_frames.append(frame)

    out_path = viz_output_path(args.video, args.arch)
    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({T} frames @ {TARGET_FPS} fps, {Q} queries)")


if __name__ == "__main__":
    main()
