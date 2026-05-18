"""Streaming inference on ScanNet-style depth estimation.

DINOv3 (frozen) + GatedMambaMix recurrent core + per-frame depth readout.
The streaming variant uses 1x8x8 patches (single-frame patches) so that the
readout can attend to a single frame at a time; temporal context comes from
the recurrent core's hidden state.

Output: {'pred_depth': (B, T, H, W)}
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.rvm_modules.rvm_wrapper import RVMStreamingDepthWrapper
from scripts_inference.utils import (
    ARCHITECTURES,
    normalization_for,
    build_architecture,
    load_checkpoint,
    load_original_frames,
    load_video_frames,
)
from scripts_inference.viz_utils import (
    colorize_depth_turbo,
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
READOUT_NUM_HEADS       = 16
DEPTH_PATCH_H           = 8
DEPTH_PATCH_W           = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tuned-ckpt", required=True)
    p.add_argument("--arch", choices=ARCHITECTURES, default="dinov3_gatedmambamix",
                   help="Model architecture (see ARCHITECTURES).")
    p.add_argument("--backbone-weights", default=None)
    p.add_argument("--out-depth", default=None, help="Optional .npy path to save the predicted depth (T, H, W).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualize", action="store_true",
                   help="Save an annotated mp4 next to the input video: each frame is the original "
                        "image and its predicted depth (Turbo colormap, per-frame normalized) "
                        "side-by-side.")
    return p.parse_args()


def build_model(arch: str, backbone_weights: str = None) -> RVMStreamingDepthWrapper:
    encoder, sequential_core = build_architecture(
        arch, SIZE, NUM_PATCHES, backbone_weights=backbone_weights,
    )
    return RVMStreamingDepthWrapper(
        encoder=encoder,
        d_enc=encoder.output_dim,
        sequential_core=sequential_core,
        freeze_encoder=True,
        readout_d_model=READOUT_D_MODEL,
        readout_num_heads=READOUT_NUM_HEADS,
        input_size=FRAME_SIZE,
        patch_h=DEPTH_PATCH_H,
        patch_w=DEPTH_PATCH_W,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading video: {args.video}")
    mean, std = normalization_for(args.arch)
    # When --visualize, stream over the WHOLE clip so the output mp4 matches the input length.
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
        out = model(frames)            # dict: {'pred_depth': (1, T, H, W)}

    pred_depth = out["pred_depth"][0].cpu().numpy()
    print(f"\nPredicted depth (streaming protocol):")
    print(f"  shape={pred_depth.shape}, min={pred_depth.min():.3f}, max={pred_depth.max():.3f}")

    if args.out_depth is not None:
        np.save(args.out_depth, pred_depth)
        print(f"  Saved to: {args.out_depth}")

    if args.visualize:
        save_visualization(args, frames_orig, pred_depth)


def save_visualization(args, frames_orig, pred_depth):
    """Save a side-by-side mp4: center-cropped original | depth heatmap.

    The model only ever sees a center-cropped square of side
    ``min(H_orig, W_orig)``. We crop the original frame to the *same*
    square before pairing it with the depth map so the two panels share
    the same field of view — the side bars of the original frame have
    no depth prediction associated with them.
    """
    T, H_orig, W_orig, _ = frames_orig.shape
    crop_side = min(H_orig, W_orig)
    top = (H_orig - crop_side) // 2
    left = (W_orig - crop_side) // 2

    out_frames = []
    for t in range(T):
        orig_cropped = frames_orig[t, top:top + crop_side, left:left + crop_side]
        d = pred_depth[t]
        d_t = torch.from_numpy(d)[None, None].float()
        d_t = F.interpolate(d_t, size=(crop_side, crop_side), mode="bilinear", align_corners=False)
        colored = colorize_depth_turbo(d_t[0, 0].numpy())
        out_frames.append(np.concatenate([orig_cropped, colored], axis=1))

    out_path = viz_output_path(args.video, args.arch)
    write_mp4(out_path, out_frames, fps=TARGET_FPS)
    print(f"[visualize] wrote {out_path}  ({T} frames, {out_frames[0].shape[1]}x{crop_side} @ {TARGET_FPS} fps)")


if __name__ == "__main__":
    main()
