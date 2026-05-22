"""Drawing + video-writing helpers for inference visualizations.

Each `inference_<task>.py` script exposes a ``--visualize`` flag that, when
set, saves an annotated mp4 next to the input video. The per-task overlays
share these primitives (text with thin outline, depth heatmap, bbox / point
drawing, mp4 muxing) which are kept here to keep ``utils.py`` focused on
model + data plumbing.

All public functions expect uint8 RGB frames (H, W, 3) and return uint8 RGB.

Output videos are encoded with H.264 via the system ``ffmpeg`` (libx264) so
that they play in browsers / VS Code's HTML5 preview. We fall back to the
OpenCV ``mp4v`` writer (MPEG-4 Part 2, not playable in Chromium) only if
ffmpeg can't be found.
"""

import shutil
import subprocess
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


# =============================================================================
# Frame conversion
# =============================================================================

def denormalize_to_uint8(
    frames: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> np.ndarray:
    """(T, 3, H, W) normalized float tensor -> (T, H, W, 3) uint8 RGB ndarray.

    Inverse of the mean/std applied in ``utils.load_video_frames``. Works for
    both ImageNet and zero-one normalizations (the latter is the identity up
    to a *255 scale).
    """
    mean_t = torch.tensor(mean, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    x = (frames.detach() * std_t + mean_t) * 255.0
    x = x.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return x


# =============================================================================
# Text drawing
# =============================================================================

def draw_text_outlined(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    fg: Tuple[int, int, int] = (0, 255, 0),
    bg: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.6,
    thickness: int = 1,
    outline_thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Draw `text` at `org` (bottom-left) with a thin dark outline.

    Mutates `frame` in place. Colors are RGB tuples. The outline is just the
    same string drawn first at a slightly thicker stroke in `bg`, then the
    foreground at the requested `thickness` on top.
    """
    cv2.putText(frame, text, org, font, font_scale, bg, outline_thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, font_scale, fg, thickness, cv2.LINE_AA)


def text_size(
    text: str,
    font_scale: float = 0.6,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> Tuple[int, int, int]:
    """Return (width, height, baseline) in pixels for cv2-rendered `text`."""
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h, baseline


# =============================================================================
# Depth colorization
# =============================================================================

def colorize_depth_turbo(depth: np.ndarray) -> np.ndarray:
    """Map a single-channel depth map to an RGB heatmap via Turbo, with close
    pixels mapped to red and far pixels to blue.

    Args:
        depth: (H, W) float ndarray. NaNs and infs are treated as the max.

    Returns:
        (H, W, 3) uint8 RGB. Normalization is per-frame min/max.
    """
    d = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(d).all():
        d = np.where(np.isfinite(d), d, np.nanmax(d[np.isfinite(d)]) if np.any(np.isfinite(d)) else 0.0)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-9:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        # Invert so the closest depth (smallest value) maps to 255 -> red end
        # of Turbo, and the farthest depth maps to 0 -> blue end.
        norm = np.clip((hi - d) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# Coordinate mapping: model-input "cropped square" -> original-frame pixels
# =============================================================================

def bbox_xyxy_to_cxcywh(bbox):
    """Convert a 4-element bbox from xyxy to cxcywh (works on list / tuple /
    torch.Tensor of shape (..., 4))."""
    if isinstance(bbox, torch.Tensor):
        x1, y1, x2, y2 = bbox.unbind(-1)
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1).abs(), (y2 - y1).abs()], dim=-1)
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)]


def bbox_cxcywh_to_xyxy(bbox):
    """Convert a 4-element bbox from cxcywh to xyxy (works on list / tuple /
    torch.Tensor of shape (..., 4))."""
    if isinstance(bbox, torch.Tensor):
        cx, cy, w, h = bbox.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    cx, cy, w, h = bbox
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def crop_square_to_orig_xy(
    u: float, v: float, H_orig: int, W_orig: int,
) -> Tuple[float, float]:
    """Map normalized (u, v) in [0, 1] of the model's center-cropped square
    input back to original-frame pixel coordinates.

    The model preprocessing resizes the short side of the frame to
    ``frame_size`` and then center-crops a ``frame_size x frame_size`` patch.
    In *original-resolution* terms this is a centered square of side
    ``min(H_orig, W_orig)``.
    """
    crop = min(H_orig, W_orig)
    left = (W_orig - crop) / 2
    top = (H_orig - crop) / 2
    return u * crop + left, v * crop + top


def crop_square_norm_to_orig_norm(
    u: float, v: float, H_orig: int, W_orig: int,
) -> Tuple[float, float]:
    """Same mapping as :func:`crop_square_to_orig_xy` but returns normalized
    coords in [0, 1] of the *original* rectangular frame (handy for the
    existing ``draw_*_norm`` helpers which take normalized inputs)."""
    x, y = crop_square_to_orig_xy(u, v, H_orig, W_orig)
    return x / W_orig, y / H_orig


# =============================================================================
# Geometry overlays
# =============================================================================

def draw_bbox_norm(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> None:
    """Draw a rectangle on `frame` from normalized xyxy in [0, 1]. Mutates in place."""
    H, W = frame.shape[:2]
    x1 = int(round(bbox_xyxy[0] * W))
    y1 = int(round(bbox_xyxy[1] * H))
    x2 = int(round(bbox_xyxy[2] * W))
    y2 = int(round(bbox_xyxy[3] * H))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label is not None:
        draw_text_outlined(frame, label, (x1, max(0, y1 - 6)), fg=color)


def draw_compass_arrow(
    frame: np.ndarray,
    center: Tuple[int, int],
    direction_xy: Tuple[float, float],
    radius: int = 18,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """Draw a compass-like rotating arrow centered at `center`.

    A circular ring marks the compass; a double-tipped arrow inside rotates
    so it points in the direction given by `direction_xy = (x, y)` in image
    coords (x rightward, y downward). The vector is L2-normalized; magnitude
    has no effect (use the accompanying text for distance).

    Mutates `frame` in place. Pass the "down" component negated if you want
    "up == forward" semantics for a top-down camera view.
    """
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(frame, (cx, cy), radius, bg, thickness + 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, color, thickness, cv2.LINE_AA)

    vx, vy = float(direction_xy[0]), float(direction_xy[1])
    mag = (vx * vx + vy * vy) ** 0.5
    if mag < 1e-9:
        # No motion -> draw a dot in the middle so the widget is still legible.
        dot_r = max(2, radius // 5)
        cv2.circle(frame, (cx, cy), dot_r, color, -1, cv2.LINE_AA)
        return
    vx, vy = vx / mag, vy / mag
    r_arrow = int(radius * 0.78)
    tip = (cx + int(round(vx * r_arrow)), cy + int(round(vy * r_arrow)))
    tail = (cx - int(round(vx * r_arrow)), cy - int(round(vy * r_arrow)))
    cv2.arrowedLine(frame, tail, tip, bg, thickness + 2, cv2.LINE_AA, tipLength=0.4)
    cv2.arrowedLine(frame, tail, tip, color, thickness, cv2.LINE_AA, tipLength=0.4)


def draw_points_norm(
    frame: np.ndarray,
    points_xy: Sequence[Tuple[float, float]],
    colors: Sequence[Tuple[int, int, int]],
    visible: Optional[Sequence[bool]] = None,
    radius: int = 4,
) -> None:
    """Draw points (filled circles) on `frame` from normalized xy in [0, 1].

    Mutates `frame` in place. `colors[i]` is the RGB color for point `i`.
    `visible[i]` False -> draw as a hollow ring instead of a filled disc.
    """
    H, W = frame.shape[:2]
    for i, (x, y) in enumerate(points_xy):
        cx = int(round(x * W))
        cy = int(round(y * H))
        col = colors[i % len(colors)]
        vis = True if visible is None else bool(visible[i])
        if vis:
            cv2.circle(frame, (cx, cy), radius, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), radius, col, 1, cv2.LINE_AA)


# =============================================================================
# Distinct color palette (Q-friendly)
# =============================================================================

# 12 visually distinct colors (RGB). Reused for query-point coloring.
PALETTE = (
    ( 64, 255,  64),  # green        — positive, neutral
    ( 64, 128, 255),  # blue         — neutral
    (255, 200,  64),  # amber
    (192,  64, 255),  # magenta
    ( 64, 255, 255),  # cyan
    (255, 128, 192),  # pink
    (128, 255, 128),  # mint
    (255, 160,  64),  # orange
    (200, 200, 255),  # lavender
    (160, 255, 200),  # seafoam
    (255, 255, 128),  # pale yellow
    # Red intentionally omitted — it reads as an error indicator.
)


# =============================================================================
# Video muxing
# =============================================================================

def write_mp4(path: str, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write a list of uint8 RGB (H, W, 3) frames to `path` as an mp4.

    Uses ``ffmpeg`` (libx264, yuv420p) when available so the output plays
    in browser-based players (VS Code preview, Chrome, GitHub). Falls back
    to the OpenCV ``mp4v`` writer if ffmpeg can't be found on PATH.
    """
    if not frames:
        raise ValueError("write_mp4: empty frame list")
    H, W = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (H, W):
            raise ValueError(f"write_mp4: frame size {f.shape[:2]} != first frame size {(H, W)}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        _write_mp4_ffmpeg(ffmpeg, path, frames, fps, W, H)
    else:
        _write_mp4_cv2(path, frames, fps, W, H)


def _write_mp4_ffmpeg(
    ffmpeg: str,
    path: str,
    frames: Sequence[np.ndarray],
    fps: float,
    W: int,
    H: int,
) -> None:
    # libx264 requires even dimensions; pad with one extra column/row if odd.
    pad_w = W % 2
    pad_h = H % 2
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", f"{float(fps):g}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        "-movflags", "+faststart",
    ]
    if pad_w or pad_h:
        cmd += ["-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd.append(path)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg returned {rc} writing {path}:\n{err}")


def _write_mp4_cv2(
    path: str,
    frames: Sequence[np.ndarray],
    fps: float,
    W: int,
    H: int,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {path}")
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def viz_output_path(video_path: str, arch: str) -> str:
    """Return ``<video_stem>.viz_<arch>.mp4`` next to the input video."""
    import os
    head, _ = os.path.splitext(video_path)
    return f"{head}.viz_{arch}.mp4"
