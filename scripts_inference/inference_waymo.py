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
    load_init_bboxes,
)
from scripts_inference.viz_utils import (
    PALETTE,
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
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
    # When --visualize, load every available frame at TARGET_FPS so a clip longer
    # than NUM_FRAMES gets a longer viz (streaming wrapper handles arbitrary T).
    n_in = None if args.visualize else NUM_FRAMES
    frames_norm = load_video_frames(args.video, n_in, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
    T_in = frames_norm.shape[0]
    init_bboxes_xyxy = load_init_bboxes(args.query).to(device)         # (Q, 4) xyxy in cropped-sq
    init_bboxes_cxcywh = bbox_xyxy_to_cxcywh(init_bboxes_xyxy)         # (Q, 4) cxcywh
    Q = init_bboxes_xyxy.shape[0]
    # Wrapper expects (B, T, C, H, W) frames and (B, 4) init. Batch by replicating
    # the same clip Q times so each query gets its own autoregressive track.
    frames = frames_norm.unsqueeze(0).expand(Q, -1, -1, -1, -1).contiguous().to(device)
    frames_orig = load_original_frames(args.video, n_in, TARGET_FPS) if args.visualize else None
    print(f"  Frames: {tuple(frames.shape)}, Q={Q} init bboxes (xyxy):")
    for q in range(Q):
        print(f"    q={q}: {init_bboxes_xyxy[q].tolist()}")

    print("Building model...")
    model = build_model(args.arch, args.backbone_weights).to(device)
    load_checkpoint(model, args.tuned_ckpt)
    model.eval()

    print("Running forward pass...")
    # Chunk encoder calls to avoid OOM on long multi-track clips
    # (Q queries * T frames at 256x256 through DINOv3-L gets heavy past ~50 frames).
    enc_batch = 32
    with torch.no_grad():
        pred_bboxes_cxcywh = model(frames, init_bboxes_cxcywh, encoder_batch_size=enc_batch)
    pred_bboxes_xyxy = bbox_cxcywh_to_xyxy(pred_bboxes_cxcywh).cpu().tolist()  # (Q, T, 4)

    print(f"\nPer-frame bbox predictions per query (streaming protocol, {T_in} frames, xyxy):")
    for q in range(Q):
        print(f"  -- query {q} --")
        for t in range(T_in):
            bb = pred_bboxes_xyxy[q][t]
            print(f"    t={t:2d}: [{bb[0]:.3f}, {bb[1]:.3f}, {bb[2]:.3f}, {bb[3]:.3f}]")

    if args.visualize:
        save_visualization(args, frames_orig, init_bboxes_xyxy.cpu().tolist(), pred_bboxes_xyxy)


def save_visualization(args, frames_orig, init_bboxes, pred_bboxes):
    """Save an annotated mp4 with per-frame predicted bboxes overlaid.

    Predictions and inits are in [0, 1] xyxy of the model's center-cropped
    square input; we remap each corner to the original rectangular frame
    for drawing. Each query gets its own color from the shared PALETTE.
    Frame 0 shows the init bbox; t>=1 shows the tracked prediction.

    Args:
        init_bboxes: (Q, 4) list of [x1, y1, x2, y2]
        pred_bboxes: (Q, T, 4) list-of-lists of [x1, y1, x2, y2]
    """
    Q = len(init_bboxes)
    T = len(pred_bboxes[0]) if Q else 0
    out_frames = []
    for t, frame in enumerate(frames_orig):
        frame = frame.copy()
        H, W = frame.shape[:2]
        thickness = max(2, min(H, W) // 200)
        font_scale = max(0.5, W / 800.0)
        for q in range(Q):
            color = PALETTE[q % len(PALETTE)]
            bbox_sq = init_bboxes[q] if t == 0 else pred_bboxes[q][t]
            x1, y1 = crop_square_norm_to_orig_norm(bbox_sq[0], bbox_sq[1], H, W)
            x2, y2 = crop_square_norm_to_orig_norm(bbox_sq[2], bbox_sq[3], H, W)
            label = f"init_{q}" if t == 0 else None
            draw_bbox_norm(frame, (x1, y1, x2, y2), color=color,
                           thickness=thickness, label=label)
        # frame counter top-left
        text_thickness = max(1, int(font_scale * 1.5))
        _, h, _ = text_size(f"t={t}", font_scale=font_scale, thickness=text_thickness)
        draw_text_outlined(frame, f"t={t}", org=(8, h + 8),
                           fg=(255, 255, 255), font_scale=font_scale, thickness=text_thickness)
        out_frames.append(frame)

    out_path = viz_output_path(args.video, args.arch)
    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({len(out_frames)} frames, {Q} tracks @ {TARGET_FPS} fps)")


if __name__ == "__main__":
    main()
