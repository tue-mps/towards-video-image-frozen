"""Streaming inference on Waymo-style object tracking.

DINOv3 (frozen) + GatedMambaMix recurrent core + per-frame bbox-conditioned
cross-attention readout. The readout attends only to the current frame's
spatial tokens at each timestep; the predicted bbox at frame t becomes the
query for frame t+1 (autoregressive tracking).

NOTE: the canonical streaming Waymo training config used Mamba (not GMMix).
For the published paper results in the streaming column of the Waymo
benchmark we keep GMMix here so the temporal core matches the offline
variant; if you want the published Mamba-only streaming setup, swap
GatedMambaMixSeqCore -> MambaSeqCore below.

Input alongside the video is a JSON sidecar with the initial bbox query:
    {"init_bbox": [x1, y1, x2, y2]}    # normalized to [0, 1]
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

from models.rvm_modules.rvm_wrapper import RVMStreamingTrackingWrapper
from scripts_inference.utils import (
    ARCHITECTURES,
    normalization_for,
    build_architecture,
    load_checkpoint,
    load_original_frames,
    load_video_frames,
    load_init_bbox,
)
from scripts_inference.viz_utils import (
    crop_square_norm_to_orig_norm,
    draw_bbox_norm,
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
TARGET_FPS   = 5
FRAME_SIZE   = 256
PATCH_SIZE   = 16
NUM_PATCHES  = (FRAME_SIZE // PATCH_SIZE) ** 2

# --- Readout ---
READOUT_D_MODEL         = 1024
READOUT_NUM_HEADS       = 4
READOUT_NUM_LAYERS      = 1
READOUT_NUM_QUERIES     = 1
BBOX_MLP_HIDDEN_DIM     = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--query", required=True, help="JSON sidecar with 'init_bbox': [x1,y1,x2,y2] in [0,1].")
    p.add_argument("--tuned-ckpt", required=True)
    p.add_argument("--arch", choices=ARCHITECTURES, default="dinov3_gatedmambamix",
                   help="Model architecture (see ARCHITECTURES).")
    p.add_argument("--backbone-weights", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated mp4 next to the input video with the per-frame predicted "
                        "bounding box overlaid. The init bbox (frame 0) is drawn in yellow; tracked "
                        "predictions for t>=1 in green.")
    return p.parse_args()


def build_model(arch: str, backbone_weights: str = None) -> RVMStreamingTrackingWrapper:
    encoder, sequential_core = build_architecture(
        arch, SIZE, NUM_PATCHES, backbone_weights=backbone_weights,
    )
    return RVMStreamingTrackingWrapper(
        encoder=encoder,
        d_enc=encoder.output_dim,
        sequential_core=sequential_core,
        d_seq_out=encoder.output_dim,
        freeze_encoder=True,
        readout_d_model=READOUT_D_MODEL,
        readout_num_heads=READOUT_NUM_HEADS,
        readout_num_layers=READOUT_NUM_LAYERS,
        readout_num_queries=READOUT_NUM_QUERIES,
        bbox_mlp_hidden_dim=BBOX_MLP_HIDDEN_DIM,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading video: {args.video}")
    mean, std = normalization_for(args.arch)
    frames_norm = load_video_frames(args.video, NUM_FRAMES, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
    frames = frames_norm.unsqueeze(0).to(device)
    frames_orig = load_original_frames(args.video, NUM_FRAMES, TARGET_FPS) if args.visualize else None
    init_bbox = load_init_bbox(args.query).to(device)
    print(f"  Frames: {tuple(frames.shape)}, init_bbox: {init_bbox.tolist()}")

    print("Building model...")
    model = build_model(args.arch, args.backbone_weights).to(device)
    load_checkpoint(model, args.tuned_ckpt)
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        pred_bboxes = model(frames, init_bbox)            # (1, T, 4)

    pred_bboxes = pred_bboxes[0].cpu().tolist()
    print(f"\nPer-frame bbox predictions (streaming protocol, {NUM_FRAMES} frames):")
    for t, bb in enumerate(pred_bboxes):
        print(f"  t={t:2d}: [{bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.3f}, {bb[3]:.3f}]")

    if args.visualize:
        save_visualization(args, frames_orig, init_bbox[0].cpu().tolist(), pred_bboxes)


def save_visualization(args, frames_orig, init_bbox, pred_bboxes):
    """Save an annotated mp4 with per-frame predicted bboxes overlaid.

    Predictions are in [0, 1] xyxy of the model's center-cropped square
    input; we remap each corner to the original rectangular frame for
    drawing. Frame 0 shows the query init_bbox in yellow; t>=1 shows the
    tracked prediction in green.
    """
    out_frames = []
    for t, frame in enumerate(frames_orig):
        frame = frame.copy()
        H, W = frame.shape[:2]
        bbox_sq = init_bbox if t == 0 else pred_bboxes[t]
        x1, y1 = crop_square_norm_to_orig_norm(bbox_sq[0], bbox_sq[1], H, W)
        x2, y2 = crop_square_norm_to_orig_norm(bbox_sq[2], bbox_sq[3], H, W)
        color = (255, 220, 64) if t == 0 else (64, 255, 64)
        label = "init" if t == 0 else None
        draw_bbox_norm(frame, (x1, y1, x2, y2), color=color, thickness=max(2, min(H, W) // 200), label=label)
        font_scale = max(0.5, W / 800.0)
        thickness = max(1, int(font_scale * 1.5))
        _, h, _ = text_size(f"t={t}", font_scale=font_scale, thickness=thickness)
        draw_text_outlined(frame, f"t={t}", org=(8, h + 8),
                           fg=(255, 255, 255), font_scale=font_scale, thickness=thickness)
        out_frames.append(frame)

    out_path = viz_output_path(args.video, args.arch)
    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({len(out_frames)} frames @ {TARGET_FPS} fps)")


if __name__ == "__main__":
    main()
