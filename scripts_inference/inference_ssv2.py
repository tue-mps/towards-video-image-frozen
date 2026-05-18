"""Streaming inference on SSv2-style action recognition.

Builds DINOv3 (frozen) + GatedMambaMix (GMMix) recurrent temporal core
+ a per-frame classification readout, loads a Lightning checkpoint, runs
the model on a video, and prints the top-K class predictions for the
final frame (the streaming protocol from the paper uses the last frame's
prediction as the video-level answer).

Edit the constants at the top of the file to switch encoder size, GMMix
depth, etc. Only paths and runtime options come from the CLI.

Example:
    python scripts_inference/inference_ssv2.py \\
        --video path/to/clip.mp4 \\
        --backbone-weights path/to/dinov3_vitl16.pth \\
        --tuned-ckpt path/to/lightning.ckpt
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.rvm_modules.rvm_wrapper import RVMStreamingClassificationWrapper
from scripts_inference.utils import (
    ARCHITECTURES,
    normalization_for,
    build_architecture,
    load_checkpoint,
    load_original_frames,
    load_video_frames,
)
from scripts_inference.viz_utils import (
    draw_text_outlined,
    text_size,
    viz_output_path,
    write_mp4,
)


# ============================================================================
# Configuration (edit here to match your trained checkpoint).
# ============================================================================

# --- Backbone size (use --arch to switch architecture) ---
SIZE = "large"                       # "small", "base", or "large"

# --- Video preprocessing ---
NUM_FRAMES   = 16
TARGET_FPS   = 6                                         # match training sampling rate
FRAME_SIZE   = 224
PATCH_SIZE   = 16
NUM_PATCHES  = (FRAME_SIZE // PATCH_SIZE) ** 2          # 196 for 224 / 16

# --- Readout (per-frame classification) ---
NUM_CLASSES             = 174              # SSv2
READOUT_NUM_PARAMS      = 768              # readout_dim
READOUT_NUM_QUERIES     = 1
READOUT_NUM_HEADS       = 12
READOUT_MLP_RATIO       = 4

# --- Output ---
TOP_K = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video file (anything decord can decode).")
    p.add_argument("--tuned-ckpt", required=True, help="Lightning .ckpt for the trained temporal core + readout.")
    p.add_argument("--arch", choices=ARCHITECTURES, default="dinov3_gatedmambamix",
                   help="Model architecture (see ARCHITECTURES).")
    p.add_argument("--backbone-weights", default=None, help="Path to DINOv3 weights. If omitted, torch.hub default is used.")
    p.add_argument("--class-names", default=None, help="Optional path to a newline-separated list of class names.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated mp4 next to the input video with the per-frame predicted "
                        "class drawn on top (green if it matches a sidecar <video>.gt.txt, else red).")
    return p.parse_args()


def build_model(arch: str, backbone_weights: str = None) -> RVMStreamingClassificationWrapper:
    encoder, sequential_core = build_architecture(
        arch, SIZE, NUM_PATCHES, backbone_weights=backbone_weights,
    )
    return RVMStreamingClassificationWrapper(
        encoder=encoder,
        d_enc=encoder.output_dim,
        sequential_core=sequential_core,
        freeze_encoder=True,
        num_classes=NUM_CLASSES,
        readout_num_params=READOUT_NUM_PARAMS,
        readout_num_queries=READOUT_NUM_QUERIES,
        readout_num_heads=READOUT_NUM_HEADS,
        readout_mlp_ratio=READOUT_MLP_RATIO,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading video: {args.video}")
    mean, std = normalization_for(args.arch)
    frames_eval = load_video_frames(args.video, NUM_FRAMES, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
    frames = frames_eval.unsqueeze(0).to(device)  # (1, T, 3, H, W)
    print(f"  Frames tensor: {tuple(frames.shape)}")

    print("Building model...")
    model = build_model(args.arch, args.backbone_weights).to(device)
    load_checkpoint(model, args.tuned_ckpt)
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        logits = model(frames, reset_state=True)  # (1, T, num_classes)

    last_frame_logits = logits[0, -1]
    probs = F.softmax(last_frame_logits, dim=-1)
    topk_probs, topk_idx = probs.topk(TOP_K)

    class_names = None
    if args.class_names is not None:
        with open(args.class_names) as f:
            class_names = [line.strip() for line in f if line.strip()]

    print(f"\nTop-{TOP_K} predictions (final-frame, streaming protocol):")
    for rank, (p, idx) in enumerate(zip(topk_probs.tolist(), topk_idx.tolist()), 1):
        name = class_names[idx] if class_names is not None and idx < len(class_names) else f"class_{idx}"
        print(f"  {rank}. {name:<40s} p={p:.4f}")

    if args.visualize:
        # For the viz pass we stream over the whole video at TARGET_FPS so the
        # output duration matches the input and we get one prediction per frame.
        # Model gets the normalized 224x224 center-crop; viz is rendered on the
        # original-resolution rectangular frames.
        print("\nRunning visualization pass over the whole video...")
        viz_frames_norm = load_video_frames(args.video, None, TARGET_FPS, FRAME_SIZE, mean=mean, std=std)
        viz_input = viz_frames_norm.unsqueeze(0).to(device)
        with torch.no_grad():
            viz_logits = model(viz_input, reset_state=True)  # (1, T_viz, num_classes)
        viz_orig = load_original_frames(args.video, None, TARGET_FPS)
        print(f"  viz tensor: {tuple(viz_input.shape)}  -> logits: {tuple(viz_logits.shape)}; "
              f"orig frames: {viz_orig.shape}")
        save_visualization(args, viz_orig, viz_logits[0], class_names)


def _load_gt_label(video_path: str) -> str:
    """Read the canonical class name from ``<video>.gt.txt`` if present."""
    gt_path = os.path.splitext(video_path)[0] + ".gt.txt"
    if not os.path.exists(gt_path):
        return None
    with open(gt_path) as f:
        return f.read().strip()


def save_visualization(args, frames_orig, logits_T, class_names):
    """Save an annotated mp4 next to the input video.

    For each frame the streaming top-1 class is drawn near the top edge,
    colored green if it matches the GT (from ``<video>.gt.txt``) and red
    otherwise. If no GT sidecar is found, all predictions are drawn in white.

    `frames_orig` is the original-resolution uint8 RGB (T, H, W, 3) ndarray;
    the model logits were produced on a 224x224 center-cropped view.
    """
    out_path = viz_output_path(args.video, args.arch)
    gt_label = _load_gt_label(args.video)
    if gt_label is None:
        print(f"\n[visualize] no <video>.gt.txt next to {args.video}; drawing all predictions in white.")
    else:
        print(f"\n[visualize] GT label: {gt_label!r}")

    probs_T = F.softmax(logits_T, dim=-1)
    top1_p, top1_idx = probs_T.max(dim=-1)
    top1_idx = top1_idx.cpu().tolist()
    top1_p = top1_p.cpu().tolist()

    out_frames = []
    for t, frame in enumerate(frames_orig):
        frame = frame.copy()
        idx = top1_idx[t]
        name = class_names[idx] if class_names is not None and idx < len(class_names) else f"class_{idx}"
        if gt_label is None:
            color = (255, 255, 255)
        else:
            color = (64, 255, 64) if name == gt_label else (255, 80, 80)
        text = f"{name}  p={top1_p[t]:.2f}"
        H, W = frame.shape[:2]
        margin = 8
        max_text_w = W - 2 * margin
        # Shrink font until the line fits the frame width (SSv2 class names get long).
        font_scale = max(0.4, W / 600.0)
        thickness = max(1, int(font_scale * 1.5))
        for _ in range(8):
            w, _, _ = text_size(text, font_scale=font_scale, thickness=thickness)
            if w <= max_text_w or font_scale <= 0.25:
                break
            font_scale *= 0.85
            thickness = max(1, int(font_scale * 1.5))
        outline = thickness + 2
        _, h, _ = text_size(text, font_scale=font_scale, thickness=thickness)
        draw_text_outlined(
            frame, text, org=(margin, h + margin),
            fg=color, font_scale=font_scale,
            thickness=thickness, outline_thickness=outline,
        )
        out_frames.append(frame)

    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({len(out_frames)} frames @ {TARGET_FPS} fps)")


if __name__ == "__main__":
    main()
