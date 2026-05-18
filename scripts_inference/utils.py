"""Shared helpers for the inference scripts.

This module exposes the four paper-released architectures and three
backbone sizes that the inference scripts can pick from, plus the shared
video-loading and checkpoint-loading routines.

Architectures (closed set):
    - "rvm"                  : original RVM (RVMViTEncoder + GatedTransformerCore)
    - "dinov3_mamba"         : frozen DINOv3 + MambaSeqCore         (paper "M")
    - "dinov3_mambamix"      : frozen DINOv3 + MambaMixSeqCore      (paper "MMix")
    - "dinov3_gatedmambamix" : frozen DINOv3 + GatedMambaMixSeqCore (paper "GMMix")

Sizes: "small", "base", "large" (RVM has no released Small checkpoint;
the architecture spec for Small mirrors DINOv3's ViT-S/16).

All paper Lightning checkpoints can be loaded via `load_checkpoint`.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_encoder.dino_wrappers import build_dinov3_encoder, DINOEncoder
from models.rvm_modules.rvm_blocks import GatedTransformerCore
from models.rvm_modules.rvm_wrapper import build_rvm_encoder, RVMViTEncoder
from models.ssm_modules.mamba_modules import (
    MambaSeqCore,
    MambaMixSeqCore,
    GatedMambaMixSeqCore,
)


# =============================================================================
# Architecture registry
# =============================================================================

ARCHITECTURES = (
    "rvm",                  # RVM ViT encoder + GatedTransformerCore
    "dinov3_rvmrnn",        # DINOv3 + GatedTransformerCore (paper "DINOv3 + RVM_RNN")
    "dinov3_mamba",         # DINOv3 + MambaSeqCore        (paper "M")
    "dinov3_mambamix",      # DINOv3 + MambaMixSeqCore     (paper "MMix")
    "dinov3_gatedmambamix", # DINOv3 + GatedMambaMixSeqCore (paper "GMMix")
)

DINOV3_NAME_BY_SIZE = {
    "small": "dinov3_vits16",
    "base":  "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

RVM_NAME_BY_SIZE = {
    "small": "rvm_vits16",
    "base":  "rvm_vitb16",
    "large": "rvm_vitl16",
}

# GatedTransformerCore head count per RVM size. For Small we use 8 heads
# (no released Small checkpoint; this is a fresh architecture spec).
RVM_CORE_HEADS_BY_SIZE = {"small": 8, "base": 12, "large": 16}

# Encoder kwargs shared by every paper-released checkpoint.
PAPER_ENCODER_KWARGS = dict(
    output_mode="all_tokens",
    mix_interm_feats=True,
    freeze=True,
)

# Mamba-family temporal-core defaults (paper Table 1 / training configs).
_MAMBA_DEFAULTS = dict(
    d_state=16,
    expand=2,
    mamba_dropout=0.1,
    chunk_size=512,
)
_SPATIAL_DEFAULTS = dict(
    spatial_num_heads=16,
    spatial_mlp_ratio=4,
)
_GATE_DEFAULTS = dict(
    gate_hidden=0,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ZERO_ONE_MEAN = (0.0, 0.0, 0.0)
ZERO_ONE_STD = (1.0, 1.0, 1.0)


def normalization_for(arch: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return (mean, std) for the architecture's expected pixel normalization.

    DINOv3-based models were trained with ImageNet normalization; RVM with
    zero-one (i.e., pixels just rescaled to [0, 1] with no further shift).
    """
    if arch == "rvm":
        return ZERO_ONE_MEAN, ZERO_ONE_STD
    return IMAGENET_MEAN, IMAGENET_STD


def load_video_frames(
    path: str,
    num_frames,
    target_fps: int,
    frame_size: int,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Decode frames at `target_fps` from `path`, resize the short side to
    `frame_size`, center-crop to `frame_size x frame_size`, and normalize
    with the given (per-channel) mean/std.

    Args:
        num_frames: int -> fixed number of frames taken from the center of
            the video (padded with the last frame if the clip is too short).
            None -> use every frame at `target_fps` from start to end (so
            the output duration matches the input duration).

    Returns a (T, 3, frame_size, frame_size) float tensor.
    """
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(path)
    total = len(vr)
    if total < 1:
        raise RuntimeError(f"Video has no frames: {path}")

    native_fps = vr.get_avg_fps() or target_fps
    step = max(1, int(round(native_fps / target_fps)))

    if num_frames is None:
        idx = list(range(0, total, step))
    else:
        span = step * (num_frames - 1)
        if span < total:
            start = (total - 1 - span) // 2
            idx = [start + i * step for i in range(num_frames)]
        else:
            idx = list(range(0, total, step))[:num_frames]
            idx = idx + [idx[-1] if idx else total - 1] * (num_frames - len(idx))

    frames = vr.get_batch(idx)                                  # (T, H, W, 3) uint8
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2).contiguous()             # (T, 3, H, W)

    _, _, H, W = frames.shape
    scale = frame_size / min(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - frame_size) // 2
    left = (new_w - frame_size) // 2
    frames = frames[:, :, top:top + frame_size, left:left + frame_size]

    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    return (frames - mean_t) / std_t


def load_original_frames(
    path: str,
    num_frames,
    target_fps: int,
) -> "np.ndarray":
    """Return the same frames as `load_video_frames` but at the source video's
    original resolution, as uint8 RGB (no resize, no center-crop, no normalize).

    Useful for visualization: the model sees a `frame_size x frame_size`
    center-crop of each frame, but we want to overlay predictions on the
    full-resolution original rectangle.

    Returns: (T, H_orig, W_orig, 3) uint8 ndarray.
    """
    import decord
    import numpy as np
    decord.bridge.set_bridge("native")

    vr = decord.VideoReader(path)
    total = len(vr)
    if total < 1:
        raise RuntimeError(f"Video has no frames: {path}")
    native_fps = vr.get_avg_fps() or target_fps
    step = max(1, int(round(native_fps / target_fps)))

    if num_frames is None:
        idx = list(range(0, total, step))
    else:
        span = step * (num_frames - 1)
        if span < total:
            start = (total - 1 - span) // 2
            idx = [start + i * step for i in range(num_frames)]
        else:
            idx = list(range(0, total, step))[:num_frames]
            idx = idx + [idx[-1] if idx else total - 1] * (num_frames - len(idx))

    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8 RGB
    return frames


def load_init_bbox(path: str) -> torch.Tensor:
    """Read a JSON sidecar with key `init_bbox` (4 numbers, normalized xyxy in [0,1]).
    Returns a (1, 4) float tensor (batch dim ready for the wrapper)."""
    import json
    with open(path) as f:
        spec = json.load(f)
    bbox = torch.tensor(spec["init_bbox"], dtype=torch.float32)
    if bbox.numel() != 4:
        raise ValueError(f"init_bbox must have 4 numbers, got {bbox.tolist()}")
    return bbox.unsqueeze(0)


def load_query_points(path: str):
    """Read a JSON sidecar with `query_points` (list of [x, y] in [0,1]) and
    optional `query_frame` (int, default 0).

    Returns ((1, Q, 2) float tensor, (1, Q) int tensor) — points and per-query
    frame index, batch-ready for `RVM*PointTracking*Wrapper`.
    """
    import json
    with open(path) as f:
        spec = json.load(f)
    pts = torch.tensor(spec["query_points"], dtype=torch.float32)
    if pts.dim() != 2 or pts.shape[1] != 2:
        raise ValueError(f"query_points must be a (Q, 2) array, got shape {tuple(pts.shape)}")
    q_frame = int(spec.get("query_frame", 0))
    Q = pts.shape[0]
    return pts.unsqueeze(0), torch.full((1, Q), q_frame, dtype=torch.long)


def _load_rvm_backbone_into(encoder: nn.Module, sequential_core: nn.Module, weights_path: str) -> None:
    """Load a converter-produced `*_pytorch_wrapper.pth` into the RVM encoder
    + GatedTransformerCore. Maps the wrapper-format keys onto our model:

      encoder.X                          -> encoder.encoder.X
      cls_token / patch_embedding.Conv_0.* -> encoder.<same>
      rvm_core.X                         -> sequential_core.X
    """
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    elif "model" in sd:
        sd = sd["model"]

    enc_state, core_state = {}, {}
    for k, v in sd.items():
        if k.startswith("rvm_core."):
            core_state[k[len("rvm_core."):]] = v
        elif k.startswith("encoder."):
            enc_state["encoder." + k[len("encoder."):]] = v
        elif k in ("cls_token", "patch_embedding.Conv_0.weight", "patch_embedding.Conv_0.bias"):
            enc_state[k] = v
        # silently ignore any other keys (readout/head/etc.)

    m_enc, u_enc = encoder.load_state_dict(enc_state, strict=False)
    m_core, u_core = sequential_core.load_state_dict(core_state, strict=False)

    # These keys are intentionally absent from the converted RVM .pth:
    #   encoder._base_posenc        : sinusoidal posenc buffer computed in __init__
    #   sequential_core.output_norm : a PyTorch-side LayerNorm with no JAX-RVM counterpart
    expected_missing_enc  = {"_base_posenc"}
    expected_missing_core = {"output_norm.weight", "output_norm.bias"}
    m_enc  = [k for k in m_enc  if k not in expected_missing_enc]
    m_core = [k for k in m_core if k not in expected_missing_core]

    print(f"Loaded RVM backbone from {weights_path}")
    print(f"  encoder: {len(enc_state)} entries, {len(m_enc)} unexpected-missing, {len(u_enc)} unexpected")
    print(f"  core:    {len(core_state)} entries, {len(m_core)} unexpected-missing, {len(u_core)} unexpected")
    if m_enc or u_enc or m_core or u_core:
        details = []
        if m_enc:   details += [f"  encoder missing: {m_enc[:5]}{'...' if len(m_enc) > 5 else ''}"]
        if u_enc:   details += [f"  encoder unexpected: {u_enc[:5]}{'...' if len(u_enc) > 5 else ''}"]
        if m_core:  details += [f"  core missing: {m_core[:5]}{'...' if len(m_core) > 5 else ''}"]
        if u_core:  details += [f"  core unexpected: {u_core[:5]}{'...' if len(u_core) > 5 else ''}"]
        raise RuntimeError(f"RVM backbone load mismatch for {weights_path!r}:\n" + "\n".join(details))


def build_architecture(
    arch: str,
    size: str,
    num_patches: int,
    backbone_weights: str = None,
) -> Tuple[nn.Module, nn.Module]:
    """Build (encoder, temporal_core) for one of the paper architectures.

    Args:
        arch: one of `ARCHITECTURES`.
        size: "small", "base", or "large".
        num_patches: spatial tokens per frame (e.g., 196 for 224 / 16).
        backbone_weights: optional path to the backbone weights. For DINOv3
            architectures this is the DINOv3 ViT `.pth`; for `arch="rvm"`
            it is the converter-produced `*_pytorch_wrapper.pth` that
            contains both the RVM ViT encoder and the GRU-gated recurrent
            core.

    Returns:
        (encoder, sequential_core) ready to plug into a task-specific wrapper.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown arch={arch!r}; expected one of {ARCHITECTURES}")

    if arch == "rvm":
        if size not in RVM_NAME_BY_SIZE:
            raise ValueError(f"Unknown size={size!r}; expected one of {sorted(RVM_NAME_BY_SIZE)}")
        encoder = build_rvm_encoder(
            model_name=RVM_NAME_BY_SIZE[size],
            output_mode="patch_tokens",
            freeze=True,
        )
        # RVM's patch_tokens output is CLS + spatial patches.
        n_tokens = num_patches + 1
        core = GatedTransformerCore(
            d_model=encoder.embed_dim,
            num_layers=4,
            num_heads=RVM_CORE_HEADS_BY_SIZE[size],
            mlp_ratio=4,
            num_patches=n_tokens,
        )
        if backbone_weights is not None:
            _load_rvm_backbone_into(encoder, core, backbone_weights)
        return encoder, core

    # DINOv3-based architectures
    if size not in DINOV3_NAME_BY_SIZE:
        raise ValueError(f"Unknown size={size!r}; expected one of {sorted(DINOV3_NAME_BY_SIZE)}")
    encoder = build_dinov3_encoder(
        model_name=DINOV3_NAME_BY_SIZE[size],
        pretrained_path=backbone_weights,
        **PAPER_ENCODER_KWARGS,
    )
    # PAPER_ENCODER_KWARGS uses output_mode="all_tokens" which emits
    # [CLS | register | patches], so the temporal core sizes its state for
    # all (1 + num_storage_tokens + spatial_patches) tokens per frame.
    n_tokens = num_patches + 1 + encoder.num_storage_tokens
    common = dict(d_model=encoder.output_dim, num_patches=n_tokens)
    if arch == "dinov3_rvmrnn":
        core = GatedTransformerCore(
            d_model=encoder.output_dim,
            num_layers=4,
            num_heads=RVM_CORE_HEADS_BY_SIZE[size],
            mlp_ratio=4,
            num_patches=n_tokens,
        )
    elif arch == "dinov3_mamba":
        core = MambaSeqCore(n_mamba_layers=4, **_MAMBA_DEFAULTS, **common)
    elif arch == "dinov3_mambamix":
        core = MambaMixSeqCore(n_layers=4, **_MAMBA_DEFAULTS, **_SPATIAL_DEFAULTS, **common)
    elif arch == "dinov3_gatedmambamix":
        core = GatedMambaMixSeqCore(
            n_layers=4, **_MAMBA_DEFAULTS, **_SPATIAL_DEFAULTS, **_GATE_DEFAULTS, **common
        )
    return encoder, core


# Prefixes saved into Lightning checkpoints that are not part of the model
# itself (e.g., learned loss-balancing weights for the Kendall-Cipolla
# pose loss). These are silently ignored when loading.
_TRAINING_ARTIFACT_PREFIXES = ("loss_fn.",)


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    """Load a Lightning .ckpt's `model.*` weights into `model`.

    The Lightning checkpoints in this codebase contain only the trained parts
    of the model (interm-feature MLPs, recurrent temporal core, readout). The
    frozen image-encoder backbone is loaded separately (DINOv3 .pth) when the
    encoder is constructed. Accordingly, this function expects:

      * every TRAINABLE model parameter to be present in the checkpoint;
      * every checkpoint key to map to a parameter in the model
        (training-artifact keys like `loss_fn.*` are dropped silently).

    Any violation raises RuntimeError with the full list of offending keys.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
        if not k.startswith(_TRAINING_ARTIFACT_PREFIXES)
    }

    model_keys = set(model.state_dict().keys())
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}

    unknown_in_ckpt = sorted(k for k in cleaned if k not in model_keys)
    missing_trainable = sorted(k for k in trainable_keys if k not in cleaned)

    model.load_state_dict(cleaned, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  {len(cleaned)} checkpoint entries -> "
          f"{len(cleaned) - len(unknown_in_ckpt)} mapped, "
          f"{len(unknown_in_ckpt)} unknown to the model")
    print(f"  {len(missing_trainable)} / {len(trainable_keys)} trainable model params missing from the checkpoint")

    if unknown_in_ckpt or missing_trainable:
        details = []
        if unknown_in_ckpt:
            details.append(f"  {len(unknown_in_ckpt)} checkpoint key(s) not present in the model:")
            details += [f"    - {k}" for k in unknown_in_ckpt]
        if missing_trainable:
            details.append(f"  {len(missing_trainable)} trainable model param(s) missing from the checkpoint:")
            details += [f"    - {k}" for k in missing_trainable]
        raise RuntimeError(
            f"Checkpoint key mismatch for {ckpt_path!r}:\n" + "\n".join(details)
        )
