"""RVM-style wrappers for all downstream tasks.

This module contains all RVM wrappers:
- RVMViTEncoder: RVM's ViT encoder (Tokenizer + sinusoidal pos embed + ViT layers)
- RVMClassificationWrapper: wraps encoder + sequential core + readout for video classification
- RVMVideoClassificationWrapper2F: 2-frame video encoder variant
- RVMTrackingWrapper / RVMTrackingWrapperPerFrame: object tracking (Waymo)
- RVMDepthWrapper / RVMDepthOnlyReadout: dense depth estimation (ScanNet)
- RVMPointTrackingWrapper / RVMPointTrackingOnlyReadout: point tracking (TAP-Vid)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict


EncoderOutputMode = Literal["patch_tokens", "cls_token"]


# =============================================================================
# Sinusoidal positional embeddings (MAE-style, matching RVM reference impl)
# =============================================================================

def get_mae_sinusoid_encoding_table(n_position, d_hid):
    """MAE-style sinusoidal positional encoding table.

    Returns: (1, n_position, d_hid) tensor.
    """
    def get_position_angle_vec(position):
        return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.tensor(sinusoid_table, dtype=torch.float32)[None, ...]


# =============================================================================
# RVM ViT Encoder
# =============================================================================

def build_rvm_encoder(
    model_name: str = "rvm_vitl16",
    output_mode: EncoderOutputMode = "patch_tokens",
    freeze: bool = True,
) -> "RVMViTEncoder":
    """Build RVM's ViT encoder for use with sequential models.

    This creates the image encoder part of RVM (patch embedding + sinusoidal
    positional encoding + ViT layers). Weight loading should be done via
    RVMClassificationWrapper.load_encoder_weights() or load_encoder_and_core_weights().

    Args:
        model_name: Model variant, currently only "rvm_vitl16" (ViT-Large, patch 16)
        output_mode: "patch_tokens" returns (B, N+1, D) including CLS,
                     "cls_token" returns (B, D) CLS only
        freeze: Whether to freeze encoder weights
    """
    RVM_CONFIGS = {
        "rvm_vits16": {"embed_dim": 384, "num_layers": 12, "num_heads": 6},
        "rvm_vitb16": {"embed_dim": 768, "num_layers": 12, "num_heads": 12},
        "rvm_vitl16": {"embed_dim": 1024, "num_layers": 24, "num_heads": 16},
    }
    if model_name not in RVM_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name}. Supported: {list(RVM_CONFIGS.keys())}")

    cfg = RVM_CONFIGS[model_name]
    encoder = RVMViTEncoder(
        embed_dim=cfg["embed_dim"],
        patch_size=16,
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4,
        output_mode=output_mode,
    )

    if freeze:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()

    encoder._frozen = freeze
    return encoder


class RVMViTEncoder(nn.Module):
    """RVM's ViT encoder (Tokenizer + sinusoidal pos embed + ViT layers, no RNN core).

    Matches the checkpoint structure from load_rvm_ckpt.py so weights can be
    loaded directly. Includes MAE-style sinusoidal positional embeddings that
    are added to patch tokens before prepending CLS (matching the original
    RVM Tokenizer behavior).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        patch_size: int = 16,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        output_mode: EncoderOutputMode = "patch_tokens",
        base_token_shape: tuple = (16, 16),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = embed_dim
        self.patch_size = patch_size
        self.output_mode = output_mode
        self.base_token_shape = base_token_shape

        # Patch embedding
        self.patch_embedding = _RVMPatchEmbedding(patch_size, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        # ViT encoder layers
        self.encoder = _RVMViTEncoderLayers(embed_dim, num_layers, num_heads, mlp_ratio)

        # Precompute base sinusoidal positional embeddings (avoid per-forward Python loop)
        bh, bw = base_token_shape
        base_posenc = get_mae_sinusoid_encoding_table(bh * bw, embed_dim)  # (1, bh*bw, D)
        self.register_buffer("_base_posenc", base_posenc.view(1, bh, bw, embed_dim))
        self._posenc_cache = {}  # (h, w) -> (1, h*w, D) tensor

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, '_frozen', False) and mode:
            super().train(False)
        return self

    def _get_sincos_posemb(self, h, w):
        """Get sinusoidal positional embeddings for (h, w) grid.

        Uses precomputed base embeddings and caches interpolated results.
        """
        key = (h, w)
        if key in self._posenc_cache:
            return self._posenc_cache[key]

        bh, bw = self.base_token_shape
        if h == bh and w == bw:
            posenc = self._base_posenc.reshape(1, h * w, self.embed_dim)
        else:
            posenc = self._base_posenc.permute(0, 3, 1, 2)  # (1, D, bh, bw)
            posenc = F.interpolate(posenc.float(), size=(h, w), mode='bicubic', align_corners=False)
            posenc = posenc.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)

        self._posenc_cache[key] = posenc
        return posenc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single images

        Returns:
            features: (B, N+1, D) patch tokens with CLS, or (B, D) CLS only
        """
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # Tokenize: (B, C, H, W) -> (B, D, h, w) -> (B, h*w, D)
        tokens = self.patch_embedding(x)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, h*w, D)

        # Add sinusoidal positional embeddings (before CLS, matching RVM reference)
        posenc = self._get_sincos_posemb(h, w).to(tokens.device, dtype=tokens.dtype)
        tokens = tokens + posenc

        # Prepend CLS token (no positional embedding for CLS, matching RVM reference)
        cls = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Encode
        encoded = self.encoder(tokens)

        if self.output_mode == "patch_tokens":
            return encoded  # (B, N+1, D) including CLS
        elif self.output_mode == "cls_token":
            return encoded[:, 0]  # (B, D)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")


class _RVMPatchEmbedding(nn.Module):
    """Patch embedding matching load_rvm_ckpt.py structure."""
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()
        self.Conv_0 = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.Conv_0(x)


class _RVMViTEncoderLayers(nn.Module):
    """ViT encoder layers matching load_rvm_ckpt.py structure."""
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([
            _RVMPreNormBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.LayerNorm_0 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm_0(x)


class _RVMPreNormBlock(nn.Module):
    """Transformer block matching load_rvm_ckpt.py structure."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = _RVMMultiHeadAttention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = _RVMTransformerMLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x_norm = self.attention_norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class _RVMMultiHeadAttention(nn.Module):
    """Multi-head attention matching load_rvm_ckpt.py structure."""
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, q, k, v, need_weights=False):
        B, N, _ = q.shape
        _, S, _ = k.shape

        q = self.query(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = self.key(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # (B, H, S, D)
        v = self.value(v).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, self.embed_dim)
        out = self.out(out)

        return out, None


class _RVMTransformerMLP(nn.Module):
    """MLP block matching load_rvm_ckpt.py structure."""
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.dense_in = nn.Linear(embed_dim, hidden_dim)
        self.dense_out = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.dense_in(x)
        x = torch.nn.functional.gelu(x)
        x = self.dense_out(x)
        return x


# =============================================================================
# Wrappers
# =============================================================================

PoolType = Literal["mean", "cls", "mlp", None]


class TokenPooler(nn.Module):
    """Pool patch tokens to a single feature vector."""

    def __init__(self, d_in: int, d_out: int, pool: PoolType = "mean"):
        super().__init__()
        self.pool = pool
        self.d_in = d_in
        self.d_out = d_out

        self.proj = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

        self.mlp = None
        if pool == "mlp":
            self.mlp = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_in),
                nn.GELU(),
                nn.Linear(d_in, d_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) - already pooled features
        returns: (B, d_out)
        """
        assert x.dim() == 2, f"TokenPooler expects (B,D), got {x.shape}"
        if self.pool == "mlp":
            return self.mlp(x)
        else:
            return self.proj(x)


class RVMClassificationWrapper(nn.Module):
    """Wrapper for video classification with RVM-style architecture.

    This wrapper is designed for video classification tasks where each video
    gets a single label. It uses:
    - A frozen image encoder (e.g., DINOv3) to extract per-frame features
    - A sequential core for temporal processing (returns sequence)
    - RVMClassificationOnlyReadout (4DS protocol) to produce classification logits

    Pipeline:
      - encoder: (B*T, C, H, W) -> (B*T, N, D_enc) patch tokens
      - reshape: (B, T, N, D_enc)
      - sequential_core: (B, T, N, D_enc) -> (B, T, N, D) temporal features
      - readout: (B, T, N, D) -> (B, num_classes) classification logits
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module,
        freeze_encoder: bool = True,
        # Readout params (RVMClassificationOnlyReadout)
        num_classes: int = 174,
        readout_num_params: int = 768,
        readout_num_queries: int = 1,
        readout_num_heads: int = 12,
        readout_num_frames: int = 16,
        readout_mlp_ratio: int = 4,
        match_vjepa_implementation: bool = True,
        add_temporal_posenc: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc

        self.sequential_core = sequential_core
        self.readout = RVMClassificationOnlyReadout(
            d_input=d_enc,
            num_classes=num_classes,
            readout_num_params=readout_num_params,
            readout_num_queries=readout_num_queries,
            readout_num_heads=readout_num_heads,
            readout_num_frames=readout_num_frames,
            readout_mlp_ratio=readout_mlp_ratio,
            match_vjepa_implementation=match_vjepa_implementation,
            add_temporal_posenc=add_temporal_posenc,
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs trainable if present
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs in train mode (for BatchNorm)
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        """All parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    def get_train_param_groups(
        self,
        weight_decay: float = 0.05,
        separate_readout: bool = True,
    ):
        """Returns optimizer param groups with proper weight decay handling."""
        decay, no_decay = [], []
        readout_decay, readout_no_decay = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_readout = name.startswith("readout.")

            # No weight decay for biases and 1D params (norm scales)
            nd = (p.ndim <= 1) or name.endswith(".bias") or ("norm" in name.lower())

            if separate_readout and is_readout:
                (readout_no_decay if nd else readout_decay).append(p)
            else:
                (no_decay if nd else decay).append(p)

        groups = []
        if decay:          groups.append({"params": decay,          "weight_decay": weight_decay})
        if no_decay:       groups.append({"params": no_decay,       "weight_decay": 0.0})
        if readout_decay:  groups.append({"params": readout_decay,  "weight_decay": weight_decay})
        if readout_no_decay:groups.append({"params": readout_no_decay,"weight_decay": 0.0})
        return groups

    # =========================================================================
    # Weight loading methods
    # =========================================================================

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into the encoder only.

        Use this when you have a pretrained image encoder (e.g., DINO, RVM ViT)
        and want to train the sequential core from scratch.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
            strict: If True, raise error on missing/unexpected keys

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        import torch
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        # Filter to encoder keys only (remove rnn_core, head, etc.)
        encoder_state = {}
        for k, v in ckpt.items():
            # Skip non-encoder keys
            if any(k.startswith(prefix) for prefix in ['rnn_core.', 'head.', 'sequential_core.', 'pooler.']):
                continue
            encoder_state[k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into both encoder and sequential_core.

        This method expects the "_pytorch_wrapper.pth" checkpoint format which has:
        - Fused Q/K/V attention weights (in_proj_weight/bias) for nn.MultiheadAttention
        - Key names matching GatedTransformerCore structure (rvm_core.transformer.layers.*)

        For loading the original RVM checkpoint, use scripts_support/load_rvm_ckpt.py
        to generate the wrapper-compatible checkpoint first.

        Args:
            checkpoint_path: Path to wrapper-compatible checkpoint (.pth)
                            Should be the "*_pytorch_wrapper.pth" file.
            strict: If True, raise error on missing/unexpected keys

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        import torch
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        # Build mapping from checkpoint to our model
        # Wrapper checkpoint has: encoder.*, cls_token, patch_embedding.*, rvm_core.*
        # Our model has: encoder.*, sequential_core.* (GatedTransformerCore used directly)
        new_state = {}

        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                # Map rvm_core.X -> sequential_core.X (strip rvm_core prefix)
                new_key = 'sequential_core.' + k[len('rvm_core.'):]
                new_state[new_key] = v
            elif k.startswith('encoder.'):
                # Checkpoint has encoder.layers.* → model needs encoder.encoder.layers.*
                # (Wrapper.encoder = RVMViTEncoder, which has .encoder = _RVMViTEncoderLayers)
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                # These are at the top level in checkpoint, need encoder. prefix
                new_key = 'encoder.' + k
                new_state[new_key] = v
            # Skip readout, head, etc.

        # Report checkpoint keys that were skipped (not mapped)
        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_full_weights(self, checkpoint_path: str, strict: bool = True):
        """Load weights into the entire model (encoder + core + readout).

        Use this for resuming training or loading a fully trained model.

        Args:
            checkpoint_path: Path to checkpoint file (.pth)
            strict: If True, raise error on missing/unexpected keys

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        import torch
        print(f"Loading full model weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        m, u = self.load_state_dict(ckpt, strict=strict)
        print(f"  Loaded full model. Missing: {len(m)}, Unexpected: {len(u)}")
        if m:
            print(f"  Missing: {m[:5]}{'...' if len(m) > 5 else ''}")
        if u:
            print(f"  Unexpected: {u[:5]}{'...' if len(u) > 5 else ''}")
        return m, u

    def forward(
        self,
        x_seq: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass for video classification.

        Args:
            x_seq: (B, T, C, H, W) - batch of video sequences
                   Automatically detects and handles both formats.
                   VideoMAE convention uses (B, C, T, H, W).
            encoder_batch_size: Optional chunk size for encoding (memory management)

        Returns:
            logits: (B, num_classes) - one prediction per video
        """
        # All datasets must return (T, C, H, W) per sample -> (B, T, C, H, W) after batching
        assert x_seq.shape[2] == 3 or x_seq.shape[2] == 1, (
            f"Expected (B, T, C, H, W) with C=3 at dim 2, got shape {x_seq.shape}. "
            f"If your dataset returns (C, T, H, W), add buffer.permute(1, 0, 2, 3) before returning."
        )

        B, T, C, H, W = x_seq.shape

        # Encode all frames
        frames_flat = x_seq.reshape(B * T, C, H, W)

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)  # (B*T, N, D_enc) or (B*T, D_enc)
        else:
            # Chunked encoding for memory efficiency
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        # Reshape to (B, T, ...)
        if tokens.dim() == 2:
            # Already pooled: (B*T, D) -> (B, T, D)
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            # Patch tokens: (B*T, N, D) -> (B, T, N, D)
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        # Temporal processing: (B, T, N, D) -> (B, T, N, D)
        tokens = self.sequential_core(tokens, return_sequence=True)

        # Classification readout: (B, T, N, D) -> (B, num_classes)
        logits = self.readout(tokens)

        return logits

    def forward_with_features(
        self,
        x_seq: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> tuple:
        """
        Forward pass that also returns intermediate features.

        Returns:
            logits: (B, num_classes)
            features: (B, T, N, D) - temporal features before readout
        """
        B, T, C, H, W = x_seq.shape

        frames_flat = x_seq.view(B * T, C, H, W).contiguous()

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        features = self.sequential_core(tokens, return_sequence=True)
        logits = self.readout(features)

        return logits, features

    def reset_state(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """Reset streaming state (if using step-by-step inference)."""
        if hasattr(self.sequential_core, 'reset_state'):
            self.sequential_core.reset_state(batch_size, device)


ClipStrategy = Literal["causal", "lookahead", "centered"]


class RVMClassificationOnlyReadout(nn.Module):
    """Classification readout following the 4DS protocol exactly.

    Uses RVMReadout (AttentionReadout) to aggregate spatio-temporal features
    into classification logits. This is the readout-only component — it expects
    pre-processed sequence features as input (no temporal core).

    Input: (B, T, N, D) sequence features from a temporal encoder
    Output: (B, num_classes) classification logits
    """

    def __init__(
        self,
        d_input: int,
        num_classes: int = 174,
        readout_num_params: int = 768,
        readout_num_queries: int = 1,
        readout_num_heads: int = 12,
        readout_num_frames: int = 16,
        readout_mlp_ratio: int = 4,
        match_vjepa_implementation: bool = True,
        add_temporal_posenc: bool = True,
    ):
        super().__init__()
        from models.rvm_modules.rvm_blocks import RVMReadout

        self.readout = RVMReadout(
            num_params=readout_num_params,
            num_classes=num_classes,
            num_queries=readout_num_queries,
            num_heads=readout_num_heads,
            num_frames=readout_num_frames,
            mlp_ratio=readout_mlp_ratio,
            input_dim=d_input,
            match_vjepa_implementation=match_vjepa_implementation,
            add_temporal_posenc=add_temporal_posenc,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: (B, T, N, D) spatio-temporal features

        Returns:
            logits: (B, num_classes)
        """
        return self.readout(sequence)


# =============================================================================
# Streaming / slot-based classification wrappers
# =============================================================================


class RVMStreamingClassificationWrapper(nn.Module):
    """Streaming per-frame classification wrapper for patch-based cores.

    Instead of producing a single label per video, this wrapper predicts a class
    label at every frame. At each timestep t, the RVMReadout attends to the
    current frame's spatial features (B, N, D) — temporal context is already
    captured by the sequential core's recurrent state.

    Pipeline:
      - encoder: (B*T, C, H, W) -> (B*T, N, D_enc)
      - reshape: (B, T, N, D_enc)
      - sequential_core: (B, T, N, D_enc) -> (B, T, N, D)
      - per-frame readout: for each t, (B, 1, N, D) -> (B, num_classes)
      - stack: (B, T, num_classes)
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module,
        freeze_encoder: bool = True,
        num_classes: int = 2,
        readout_num_params: int = 768,
        readout_num_queries: int = 1,
        readout_num_heads: int = 12,
        readout_mlp_ratio: int = 4,
        match_vjepa_implementation: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc
        self.sequential_core = sequential_core
        self._seq_state = None  # For streaming state carryover

        from models.rvm_modules.rvm_blocks import RVMReadout
        self.readout = RVMReadout(
            num_params=readout_num_params,
            num_classes=num_classes,
            num_queries=readout_num_queries,
            num_heads=readout_num_heads,
            num_frames=1,
            mlp_ratio=readout_mlp_ratio,
            input_dim=d_enc,
            match_vjepa_implementation=match_vjepa_implementation,
            add_temporal_posenc=False,
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs trainable if present
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs in train mode (for BatchNorm)
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rnn_core.', 'rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.']):
                continue
            encoder_state[k] = v
        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def _encode_frames(self, x_seq: torch.Tensor, encoder_batch_size: Optional[int] = None) -> torch.Tensor:
        """Batch-encode all frames: (B, T, C, H, W) -> (B, T, N, D)."""
        B, T, C, H, W = x_seq.shape
        frames_flat = x_seq.reshape(B * T, C, H, W)

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunks.append(self.encoder(chunk))
            tokens = torch.cat(tokens_chunks, dim=0)

        N, D = tokens.shape[1], tokens.shape[2]
        return tokens.view(B, T, N, D)

    def _per_frame_readout(self, core_out: torch.Tensor) -> torch.Tensor:
        """Per-frame readout: (B, T, N, D) -> (B, T, num_classes)."""
        B, T = core_out.shape[:2]
        per_frame_logits = []
        for t in range(T):
            frame_feats = core_out[:, t:t+1]               # (B, 1, N, D)
            logits_t = self.readout(frame_feats)            # (B, num_classes)
            per_frame_logits.append(logits_t)
        return torch.stack(per_frame_logits, dim=1)         # (B, T, num_classes)

    def forward(
        self,
        x_seq: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
        reset_state: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, C, H, W) video frames
            reset_state: If True, initialize fresh recurrent state (default, training).
                         If False, continue from previous call's state (streaming eval).

        Returns:
            logits: (B, T, num_classes)
        """
        # All datasets must return (T, C, H, W) per sample -> (B, T, C, H, W) after batching
        assert x_seq.shape[2] == 3 or x_seq.shape[2] == 1, (
            f"Expected (B, T, C, H, W) with C=3 at dim 2, got shape {x_seq.shape}. "
            f"If your dataset returns (C, T, H, W), add buffer.permute(1, 0, 2, 3) before returning."
        )
        tokens = self._encode_frames(x_seq, encoder_batch_size)
        B, T = tokens.shape[:2]

        if hasattr(self.sequential_core, 'forward_sequence'):
            core_out = self.sequential_core.forward_sequence(tokens)  # (B, T, N, D)
        else:
            core_out = self.sequential_core(tokens, return_sequence=True)  # (B, T, N, D)

        return self._per_frame_readout(core_out)


class FourierPositionEncoding(nn.Module):
    """Fourier feature encoding for continuous coordinates.

    Maps each coordinate to sin/cos features at multiple frequencies,
    following the positional encoding approach used in 4DS/NeRF.
    """

    def __init__(self, num_frequencies: int = 16, input_dim: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        # Output: input_dim * num_frequencies * 2 (sin + cos)
        self.output_dim = input_dim * num_frequencies * 2

        # Log-spaced frequencies from 2^0 to 2^(num_freq-1)
        freqs = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freqs', freqs)  # (num_frequencies,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) coordinates in [0, 1]
        Returns:
            features: (B, input_dim * num_frequencies * 2)
        """
        # x: (B, D) -> (B, D, 1) * (1, 1, F) -> (B, D, F)
        x_freq = x.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0) * math.pi
        # Concatenate sin and cos: (B, D, F) -> (B, D*F*2)
        features = torch.cat([x_freq.sin(), x_freq.cos()], dim=-1)  # (B, D, 2F)
        return features.reshape(x.shape[0], -1)  # (B, D*2F)


# =============================================================================
# Object Tracking (Waymo)
# =============================================================================

class BBoxQueryEncoder(nn.Module):
    """Encode bounding box coordinates into query tokens for cross-attention.

    Following 4DS paper: bbox positions are embedded using Fourier features
    (16 bases) and passed through an MLP with 512 hidden and output units
    before being used as queries for cross-attention.
    """

    def __init__(self, d_model: int = 768, num_queries: int = 1,
                 num_fourier_frequencies: int = 16, mlp_hidden_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.mlp_hidden_dim = mlp_hidden_dim

        # Fourier encoding for bbox [cx, cy, w, h]
        self.fourier_enc = FourierPositionEncoding(
            num_frequencies=num_fourier_frequencies, input_dim=4
        )
        fourier_dim = self.fourier_enc.output_dim  # 4 * num_freq * 2 = 128

        # 4DS: MLP with 512 hidden and 512 output units
        self.bbox_encoder = nn.Sequential(
            nn.Linear(fourier_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        # Project from mlp_hidden_dim to d_model for cross-attention
        self.to_query = nn.Linear(mlp_hidden_dim, d_model)

        # Generate multiple queries from single bbox encoding
        if num_queries > 1:
            self.query_proj = nn.Linear(d_model, d_model * num_queries)
        else:
            self.query_proj = None

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (B, 4) initial bounding box [cx, cy, w, h]

        Returns:
            queries: (B, num_queries, d_model) query tokens
        """
        # Fourier encode bbox coordinates
        fourier_features = self.fourier_enc(bbox)  # (B, fourier_dim=128)
        encoded = self.bbox_encoder(fourier_features)  # (B, mlp_hidden_dim=512)
        queries = self.to_query(encoded)  # (B, d_model)

        if self.query_proj is not None:
            queries = self.query_proj(queries)  # (B, d_model * num_queries)
            queries = queries.view(-1, self.num_queries, self.d_model)  # (B, Q, d_model)
        else:
            queries = queries.unsqueeze(1)  # (B, 1, d_model)

        return queries


class TrackingCrossAttentionReadout(nn.Module):
    """Cross-attention readout for tracking that produces per-frame bbox predictions.

    Following 4DS paper: single cross-attention layer with 1024 channels and 4 heads.
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 4,
        num_layers: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * mlp_ratio,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
            )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, sequence)
        return self.output_norm(query)


class RVMTrackingWrapper(nn.Module):
    """Wrapper for object tracking with RVM-style architecture.

    Architecture (following RVM paper Section 11.1):
    - Frozen image encoder extracts per-frame features
    - Sequential core processes temporal sequence
    - Bbox-conditioned cross-attention readout predicts per-frame bboxes
    - Uses 1024 channels (internally) and 4 heads for readout
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module,
        d_seq_out: int,
        freeze_encoder: bool = True,
        readout_d_model: int = 1024,
        readout_num_heads: int = 4,
        readout_num_layers: int = 1,
        readout_num_queries: int = 1,
        bbox_mlp_hidden_dim: int = 512,
        num_frames: int = 16,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc
        self.num_frames = num_frames

        self.sequential_core = sequential_core

        self.proj_to_readout = nn.Linear(d_enc, readout_d_model) if d_enc != readout_d_model else nn.Identity()

        self.bbox_query_encoder = BBoxQueryEncoder(
            d_model=readout_d_model,
            num_queries=readout_num_queries,
            mlp_hidden_dim=bbox_mlp_hidden_dim,
        )

        self.readout = TrackingCrossAttentionReadout(
            d_model=readout_d_model,
            num_heads=readout_num_heads,
            num_layers=readout_num_layers,
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(readout_d_model * readout_num_queries, readout_d_model),
            nn.GELU(),
            nn.Linear(readout_d_model, num_frames * 4),
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs trainable if present
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs in train mode (for BatchNorm)
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into the encoder only."""
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.', 'bbox_']):
                continue
            encoder_state[k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into both encoder and sequential_core."""
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                # Map rvm_core.X -> sequential_core.X (strip rvm_core prefix)
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def forward(
        self,
        frames: torch.Tensor,
        init_bbox: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        if hasattr(self.sequential_core, 'forward_sequence'):
            tokens = self.sequential_core.forward_sequence(tokens)
        else:
            tokens = self.sequential_core(tokens, return_sequence=True)

        if tokens.dim() == 4:
            B, T, N, D = tokens.shape
            sequence = tokens.reshape(B, T * N, D)
        else:
            sequence = tokens

        sequence = self.proj_to_readout(sequence)
        queries = self.bbox_query_encoder(init_bbox)
        output = self.readout(queries, sequence)

        output_flat = output.reshape(B, -1)
        pred_bboxes = self.bbox_head(output_flat)
        pred_bboxes = pred_bboxes.reshape(B, self.num_frames, 4)
        return pred_bboxes


class RVMStreamingTrackingWrapper(nn.Module):
    """Streaming per-frame tracking wrapper.

    Instead of aggregating all T frames into a single readout, this wrapper
    predicts bounding boxes frame-by-frame. At each timestep t, cross-attention
    reads only the current frame's spatial features (B, N, D) — temporal context
    is already captured by the sequential core's recurrent state.

    The predicted bbox at frame t is re-encoded as the query for frame t+1,
    creating an autoregressive tracking signal.

    Architecture:
    - Frozen encoder extracts per-frame features
    - Sequential core processes temporal sequence (parallel for training)
    - Per-frame: cross-attention readout on (B, N, D) -> bbox_head -> 4 coords
    - Autoregressive: predicted bbox -> new query for next frame

    Output: (B, T, 4).
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module,
        d_seq_out: int,
        freeze_encoder: bool = True,
        readout_d_model: int = 1024,
        readout_num_heads: int = 4,
        readout_num_layers: int = 1,
        readout_num_queries: int = 1,
        bbox_mlp_hidden_dim: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc

        self.sequential_core = sequential_core

        self.proj_to_readout = nn.Linear(d_enc, readout_d_model) if d_enc != readout_d_model else nn.Identity()

        self.bbox_query_encoder = BBoxQueryEncoder(
            d_model=readout_d_model,
            num_queries=readout_num_queries,
            mlp_hidden_dim=bbox_mlp_hidden_dim,
        )

        self.readout = TrackingCrossAttentionReadout(
            d_model=readout_d_model,
            num_heads=readout_num_heads,
            num_layers=readout_num_layers,
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(readout_d_model * readout_num_queries, readout_d_model),
            nn.GELU(),
            nn.Linear(readout_d_model, 4),
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs trainable if present
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs in train mode (for BatchNorm)
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into the encoder only."""
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.', 'bbox_']):
                continue
            encoder_state[k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into both encoder and sequential_core."""
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def forward(
        self,
        frames: torch.Tensor,
        init_bbox: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # 1. Encode all frames (batched)
        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        # 2. Sequential core (parallel full-sequence processing)
        if hasattr(self.sequential_core, 'forward_sequence'):
            tokens = self.sequential_core.forward_sequence(tokens)
        else:
            tokens = self.sequential_core(tokens, return_sequence=True)

        # 3. Project to readout dimension
        # tokens: (B, T, N, D) or (B, T, D)
        tokens = self.proj_to_readout(tokens)

        # 4. Per-frame autoregressive readout
        query = self.bbox_query_encoder(init_bbox)  # (B, Q, d_readout)

        pred_bboxes = []
        for t in range(T):
            if tokens.dim() == 4:
                frame_tokens = tokens[:, t, :, :]  # (B, N, d_readout)
            else:
                frame_tokens = tokens[:, t:t+1, :]  # (B, 1, d_readout)

            query = self.readout(query, frame_tokens)  # (B, Q, d_readout)
            bbox_t = self.bbox_head(query.reshape(B, -1))  # (B, 4)
            pred_bboxes.append(bbox_t)

            # Autoregressive: re-encode predicted bbox as next query
            query = self.bbox_query_encoder(bbox_t.detach())  # (B, Q, d_readout)

        return torch.stack(pred_bboxes, dim=1)  # (B, T, 4)


class RVMDepthOnlyReadout(nn.Module):
    """Depth estimation readout following the 4DS protocol exactly.

    Uses spatio-temporal 2x8x8 patches with RVMReadout (AttentionReadout).
    Each query predicts 128 depth values (one per pixel in its patch).
    """

    def __init__(
        self,
        d_input: int,
        readout_d_model: int = 1024,
        readout_num_heads: int = 16,
        num_frames: int = 16,
        input_size: int = 224,
        patch_t: int = 2,
        patch_h: int = 8,
        patch_w: int = 8,
    ):
        super().__init__()
        from models.rvm_modules.rvm_blocks import RVMReadout

        self.num_frames = num_frames
        self.input_size = input_size
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.nt = num_frames // patch_t
        self.nh = input_size // patch_h
        self.nw = input_size // patch_w
        self.num_queries = self.nt * self.nh * self.nw

        self.num_classes = patch_t * patch_h * patch_w

        self.readout = RVMReadout(
            num_params=readout_d_model,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            num_heads=readout_num_heads,
            num_frames=num_frames,
            mlp_ratio=4,
            input_dim=d_input,
            match_vjepa_implementation=True,
            add_temporal_posenc=True,
        )

    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = sequence.shape[0]

        out = self.readout(sequence)

        out = out.view(B, self.nt, self.nh, self.nw, self.patch_t, self.patch_h, self.patch_w)
        out = out.permute(0, 1, 4, 2, 5, 3, 6)
        out = out.reshape(B,
                          self.nt * self.patch_t,
                          self.nh * self.patch_h,
                          self.nw * self.patch_w)

        return {'pred_depth': out}


class RVMStreamingDepthWrapper(nn.Module):
    """Streaming per-frame depth estimation wrapper.

    Same encoder + sequential core pipeline as RVMDepthWrapper, but the readout
    attends to each frame's spatial tokens independently (per-frame) instead of
    all T*N tokens at once. Uses spatial-only patches (patch_t=1) instead of
    spatio-temporal patches (patch_t=2).
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module = None,
        freeze_encoder: bool = True,
        readout_d_model: int = 1024,
        readout_num_heads: int = 16,
        input_size: int = 224,
        patch_h: int = 8,
        patch_w: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc
        self.sequential_core = sequential_core
        self.freeze_encoder = freeze_encoder

        # Per-frame readout: num_frames=1, patch_t=1
        self.readout = RVMDepthOnlyReadout(
            d_input=d_enc,
            readout_d_model=readout_d_model,
            readout_num_heads=readout_num_heads,
            num_frames=1,
            input_size=input_size,
            patch_t=1,
            patch_h=patch_h,
            patch_w=patch_w,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        frames: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        if self.sequential_core is not None:
            if hasattr(self.sequential_core, 'forward_sequence'):
                tokens = self.sequential_core.forward_sequence(tokens)
            else:
                result = self.sequential_core(tokens, return_sequence=True)
                tokens = result if isinstance(result, torch.Tensor) else result[0]

        # Per-frame readout
        pred_depths = []
        for t in range(T):
            frame_tokens = tokens[:, t:t+1, :, :]  # (B, 1, N, D)
            result = self.readout(frame_tokens)      # {'pred_depth': (B, 1, H, W)}
            pred_depths.append(result['pred_depth'])
        pred_depth = torch.cat(pred_depths, dim=1)   # (B, T, H, W)

        return {'pred_depth': pred_depth}

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in [
                'rvm_core.', 'head.', 'sequential_core.', 'pooler.',
                'readout.', 'bbox_', 'decoder.', 'query_encoder.',
            ]):
                continue
            encoder_state[k] = v
        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")
        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)
        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v
        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")
        m, u = self.load_state_dict(new_state, strict=strict)
        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u


# =============================================================================
# Point Tracking (TAP-Vid / Perception Test)
# =============================================================================

class PointQueryEncoder4DS(nn.Module):
    """Encode point coordinates into query tokens following 4DS paper exactly.

    4DS paper: "Fourier features (with 16 bases)" followed by
    "MLP with 512 hidden and output units"

    Then: "replicate the query for a track 8 times (adding learnable temporal
    embeddings) and predict 2 frames at a time from each query"
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_temporal_queries: int = 8,
        num_fourier_frequencies: int = 16,
        mlp_hidden_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_temporal_queries = num_temporal_queries
        self.mlp_hidden_dim = mlp_hidden_dim

        self.fourier_enc = FourierPositionEncoding(
            num_frequencies=num_fourier_frequencies, input_dim=2
        )
        fourier_dim = self.fourier_enc.output_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(fourier_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )

        self.to_query = nn.Linear(mlp_hidden_dim, d_model)

        self.temporal_embed = nn.Parameter(
            torch.randn(1, num_temporal_queries, d_model) * 0.02
        )

    def forward(self, query_point: torch.Tensor) -> torch.Tensor:
        B = query_point.shape[0]
        fourier_features = self.fourier_enc(query_point)
        encoded = self.point_mlp(fourier_features)
        query = self.to_query(encoded)

        queries = query.unsqueeze(1).expand(-1, self.num_temporal_queries, -1)
        queries = queries + self.temporal_embed.expand(B, -1, -1)
        return queries


class PointTrackingReadout4DS(nn.Module):
    """Cross-attention readout for point tracking following 4DS paper.

    4DS paper: "cross-attention layer with 1024 channels and 8 heads"
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * mlp_ratio,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
            )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, sequence)
        return self.output_norm(query)


class RVMPointTrackingOnlyReadout(nn.Module):
    """Point tracking readout module following 4DS paper exactly.

    4DS protocol:
    - Query encoding: Fourier (16 bases) + MLP (512 hidden/output)
    - 8 temporal queries with learnable temporal embeddings
    - Cross-attention: 1024 channels, 8 heads
    - Predict 2 frames per query: position (x, y), visibility, uncertainty
    - Total: 8 queries x 2 frames = 16 frames output
    """

    def __init__(
        self,
        d_input: int,
        readout_d_model: int = 1024,
        readout_num_heads: int = 8,
        readout_num_layers: int = 1,
        num_temporal_queries: int = 8,
        frames_per_query: int = 2,
        num_frames: int = 16,
        num_fourier_frequencies: int = 16,
        mlp_hidden_dim: int = 512,
        predict_uncertainty: bool = True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_temporal_queries = num_temporal_queries
        self.frames_per_query = frames_per_query
        self.predict_uncertainty = predict_uncertainty
        self.d_model = readout_d_model

        self.proj_to_readout = nn.Linear(d_input, readout_d_model) if d_input != readout_d_model else nn.Identity()

        self.point_query_encoder = PointQueryEncoder4DS(
            d_model=readout_d_model,
            num_temporal_queries=num_temporal_queries,
            num_fourier_frequencies=num_fourier_frequencies,
            mlp_hidden_dim=mlp_hidden_dim,
        )

        self.readout = PointTrackingReadout4DS(
            d_model=readout_d_model,
            num_heads=readout_num_heads,
            num_layers=readout_num_layers,
        )

        output_per_query = frames_per_query * (2 + 1 + (1 if predict_uncertainty else 0))
        self.prediction_head = nn.Sequential(
            nn.Linear(readout_d_model, readout_d_model // 2),
            nn.GELU(),
            nn.Linear(readout_d_model // 2, output_per_query),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        query_point: torch.Tensor,
        query_frame: Optional[torch.Tensor] = None,
    ) -> dict:
        B = sequence.shape[0]

        if sequence.dim() == 4:
            B, T, N, D = sequence.shape
            sequence = sequence.reshape(B, T * N, D)

        sequence = self.proj_to_readout(sequence)
        queries = self.point_query_encoder(query_point)
        output = self.readout(queries, sequence)
        predictions = self.prediction_head(output)

        predictions = predictions.reshape(B, self.num_frames, -1)

        pred_points = torch.sigmoid(predictions[:, :, :2])

        if query_frame is None:
            pred_points = pred_points.clone()
            pred_points[:, 0, :] = query_point
        else:
            pred_points = pred_points.clone()
            for b in range(B):
                qf = query_frame[b].item()
                if 0 <= qf < self.num_frames:
                    pred_points[b, qf, :] = query_point[b]

        pred_visibility_logits = predictions[:, :, 2].clone()

        if query_frame is None:
            pred_visibility_logits[:, 0] = 10.0
        else:
            for b in range(B):
                qf = query_frame[b].item()
                if 0 <= qf < self.num_frames:
                    pred_visibility_logits[b, qf] = 10.0

        result = {
            'pred_points': pred_points,
            'pred_visibility': pred_visibility_logits,
        }

        if self.predict_uncertainty:
            pred_uncertainty_logits = predictions[:, :, 3].clone()
            if query_frame is None:
                pred_uncertainty_logits[:, 0] = -10.0
            else:
                for b in range(B):
                    qf = query_frame[b].item()
                    if 0 <= qf < self.num_frames:
                        pred_uncertainty_logits[b, qf] = -10.0
            result['pred_uncertainty'] = pred_uncertainty_logits

        return result


class RVMPointTrackingWrapper(nn.Module):
    """Encoder + optional sequential core + readout for point tracking.

    Follows the same structure as other RVM wrappers:
    - self.encoder: frozen image encoder (ViT)
    - self.sequential_core: optional temporal sequence model
    - self.readout: task-specific point tracking readout head

    Supports both single-query (B, 2) and multi-query (B, Q, 2) inputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        readout: nn.Module,
        sequential_core: nn.Module = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc
        self.readout = readout
        self.sequential_core = sequential_core

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs trainable if present
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            # Keep intermediate feature mixing MLPs in train mode (for BatchNorm)
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into the encoder only."""
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.', 'bbox_']):
                continue
            encoder_state[k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into both encoder and sequential_core."""
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def forward(
        self,
        frames: torch.Tensor,
        query_point: torch.Tensor,
        query_frame: torch.Tensor = None,
        encoder_batch_size: Optional[int] = None,
    ) -> dict:
        """
        Args:
            frames: (B, T, C, H, W)
            query_point: (B, 2) single query or (B, Q, 2) multi-query
            query_frame: (B,) or (B, Q) matching query_point shape
            encoder_batch_size: optional chunked encoding for memory efficiency
        Returns:
            dict with predictions shaped (B, T, ...) or (B, Q, T, ...)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # Encode frames (with optional chunking for memory efficiency)
        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        # Reshape: (B*T, ...) → (B, T, ...)
        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        # Apply sequential core if present
        if self.sequential_core is not None:
            if hasattr(self.sequential_core, 'forward_sequence'):
                tokens = self.sequential_core.forward_sequence(tokens)
            else:
                result = self.sequential_core(tokens, return_sequence=True)
                tokens = result if isinstance(result, torch.Tensor) else result[0]

        # Multi-query: query_point is (B, Q, 2)
        if query_point.dim() == 3:
            Q = query_point.shape[1]
            # Expand tokens: (B, T, ...) → (B*Q, T, ...)
            if tokens.dim() == 4:
                tokens_exp = tokens.unsqueeze(1).expand(-1, Q, -1, -1, -1)
                tokens_exp = tokens_exp.reshape(B * Q, T, tokens.shape[2], tokens.shape[3])
            else:  # (B, T, D)
                tokens_exp = tokens.unsqueeze(1).expand(-1, Q, -1, -1)
                tokens_exp = tokens_exp.reshape(B * Q, T, tokens.shape[2])

            qp_flat = query_point.reshape(B * Q, 2)
            qf_flat = query_frame.reshape(B * Q) if query_frame is not None else None

            output = self.readout(tokens_exp, qp_flat, qf_flat)

            # Reshape back: (B*Q, ...) → (B, Q, ...)
            result = {}
            for k, v in output.items():
                result[k] = v.reshape(B, Q, *v.shape[1:])
            return result
        else:
            # Single query: (B, 2)
            return self.readout(tokens, query_point, query_frame)


class RVMStreamingPointTrackingWrapper(nn.Module):
    """Streaming per-frame point tracking wrapper.

    Same as RVMPointTrackingWrapper but with per-frame readout:
    - Encoder and sequential core process the full sequence (same as standard).
    - Readout attends to each frame's spatial tokens independently (N tokens
      instead of T*N), predicting position/visibility/uncertainty per frame.
    - Temporal context comes from the sequential core's recurrent state,
      NOT from cross-attention over multiple frames.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        readout_d_model: int = 1024,
        readout_num_heads: int = 8,
        readout_num_layers: int = 1,
        num_fourier_frequencies: int = 16,
        mlp_hidden_dim: int = 512,
        predict_uncertainty: bool = True,
        sequential_core: nn.Module = None,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc
        self.sequential_core = sequential_core
        self.predict_uncertainty = predict_uncertainty
        self.d_model = readout_d_model

        # Project encoder tokens to readout dimension
        self.proj_to_readout = nn.Linear(d_enc, readout_d_model) if d_enc != readout_d_model else nn.Identity()

        # Query encoder: Fourier + MLP → single query (no temporal embeddings)
        self.fourier_enc = FourierPositionEncoding(
            num_frequencies=num_fourier_frequencies, input_dim=2,
        )
        fourier_dim = self.fourier_enc.output_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(fourier_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
        )
        self.to_query = nn.Linear(mlp_hidden_dim, readout_d_model)

        # Per-frame cross-attention readout
        self.readout = PointTrackingReadout4DS(
            d_model=readout_d_model,
            num_heads=readout_num_heads,
            num_layers=readout_num_layers,
        )

        # Prediction head: per-frame output (position + visibility + uncertainty)
        out_dim = 2 + 1 + (1 if predict_uncertainty else 0)
        self.prediction_head = nn.Sequential(
            nn.Linear(readout_d_model, readout_d_model // 2),
            nn.GELU(),
            nn.Linear(readout_d_model // 2, out_dim),
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.', 'bbox_']):
                continue
            encoder_state[k] = v
        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")
        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)
        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v
        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")
        m, u = self.load_state_dict(new_state, strict=strict)
        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def _encode_query(self, query_point: torch.Tensor) -> torch.Tensor:
        """Encode query point into a single query vector: (B, 2) → (B, 1, d_model)."""
        fourier_features = self.fourier_enc(query_point)
        encoded = self.point_mlp(fourier_features)
        query = self.to_query(encoded)
        return query.unsqueeze(1)  # (B, 1, d_model)

    def forward(
        self,
        frames: torch.Tensor,
        query_point: torch.Tensor,
        query_frame: torch.Tensor = None,
        encoder_batch_size: Optional[int] = None,
    ) -> dict:
        """
        Args:
            frames: (B, T, C, H, W)
            query_point: (B, 2) single query or (B, Q, 2) multi-query
            query_frame: (B,) or (B, Q)
        Returns:
            dict with 'pred_points' (B, T, 2), 'pred_visibility' (B, T), etc.
            or (B, Q, T, ...) for multi-query
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # Encode frames
        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        # Reshape: (B*T, N, D) → (B, T, N, D)
        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        # Sequential core
        if self.sequential_core is not None:
            if hasattr(self.sequential_core, 'forward_sequence'):
                tokens = self.sequential_core.forward_sequence(tokens)
            else:
                result = self.sequential_core(tokens, return_sequence=True)
                tokens = result if isinstance(result, torch.Tensor) else result[0]

        # Multi-query handling: batch Q queries into the batch dimension
        if query_point.dim() == 3:
            Q = query_point.shape[1]
            # Expand tokens: (B, T, N, D) → (B*Q, T, N, D)
            if tokens.dim() == 4:
                tokens_exp = tokens.unsqueeze(1).expand(-1, Q, -1, -1, -1)
                tokens_exp = tokens_exp.reshape(B * Q, T, tokens.shape[2], tokens.shape[3])
            else:  # (B, T, D)
                tokens_exp = tokens.unsqueeze(1).expand(-1, Q, -1, -1)
                tokens_exp = tokens_exp.reshape(B * Q, T, tokens.shape[2])
            # Flatten queries: (B, Q, 2) → (B*Q, 2)
            qp_flat = query_point.reshape(B * Q, 2)
            qf_flat = query_frame.reshape(B * Q) if query_frame is not None else None
            # Single call for all queries
            output = self._forward_single_query(tokens_exp, qp_flat, qf_flat, T)
            # Reshape back: (B*Q, T, ...) → (B, Q, T, ...)
            return {k: v.reshape(B, Q, *v.shape[1:]) for k, v in output.items()}
        else:
            return self._forward_single_query(tokens, query_point, query_frame, T)

    def _forward_single_query(self, tokens, query_point, query_frame, T):
        """Per-frame readout for a single query point (batched over T frames)."""
        B = tokens.shape[0]

        # Project to readout dimension: (B, T, N, D) → (B, T, N, d_model)
        if tokens.dim() == 4:
            N = tokens.shape[2]
            tokens_proj = self.proj_to_readout(tokens)  # (B, T, N, d_model)
        else:
            # (B, T, D) — no spatial dim
            tokens_proj = self.proj_to_readout(tokens).unsqueeze(2)  # (B, T, 1, d_model)
            N = 1

        # Encode query: (B, 1, d_model)
        query = self._encode_query(query_point)

        # Batch all T frames into the batch dimension for a single readout call
        tokens_flat = tokens_proj.reshape(B * T, N, self.d_model)          # (B*T, N, d_model)
        query_exp = query.unsqueeze(1).expand(-1, T, -1, -1)              # (B, T, 1, d_model)
        query_flat = query_exp.reshape(B * T, 1, self.d_model)            # (B*T, 1, d_model)

        out = self.readout(query_flat, tokens_flat)                        # (B*T, 1, d_model)
        pred = self.prediction_head(out.squeeze(1))                        # (B*T, out_dim)
        pred = pred.reshape(B, T, -1)                                      # (B, T, out_dim)

        pred_points = torch.sigmoid(pred[:, :, :2])                        # (B, T, 2)
        pred_visibility_logits = pred[:, :, 2]                             # (B, T)

        # Force query frame prediction to match ground truth (vectorized)
        pred_points = pred_points.clone()
        pred_visibility_logits = pred_visibility_logits.clone()

        if query_frame is None:
            pred_points[:, 0, :] = query_point.to(pred_points.dtype)
            pred_visibility_logits[:, 0] = 10.0
        else:
            batch_idx = torch.arange(B, device=query_frame.device)
            qf = query_frame.long()
            valid = (qf >= 0) & (qf < T)
            b_v = batch_idx[valid]
            t_v = qf[valid]
            pred_points[b_v, t_v, :] = query_point[valid].to(pred_points.dtype)
            pred_visibility_logits[b_v, t_v] = 10.0

        result = {
            'pred_points': pred_points,
            'pred_visibility': pred_visibility_logits,
        }

        if self.predict_uncertainty:
            pred_uncertainty_logits = pred[:, :, 3].clone()                # (B, T)
            if query_frame is None:
                pred_uncertainty_logits[:, 0] = -10.0
            else:
                pred_uncertainty_logits[b_v, t_v] = -10.0
            result['pred_uncertainty'] = pred_uncertainty_logits

        return result


# =============================================================================
# Slot-based Object Tracking (for MooG V4-V6 cores)
# =============================================================================

class RVMStreamingCameraPoseWrapper(nn.Module):
    """Streaming per-frame camera pose estimation wrapper.

    Predicts frame-to-frame camera pose deltas in a streaming fashion.
    At each timestep t, the sequential core's recurrent state captures
    temporal context, and a lightweight MLP head predicts the 9D pose delta
    (3D translation + 6D rotation representation) from pooled frame features.

    The 6D rotation representation follows Zhou et al. (CVPR 2019) —
    "On the Continuity of Rotation Representations in Neural Networks" —
    which avoids quaternion discontinuities.

    Output per frame: 9D vector = [dx, dy, dz, r1, r2, r3, r4, r5, r6]
      - (dx, dy, dz): translation delta
      - (r1..r6): 6D rotation (first two columns of rotation matrix,
        orthogonalized via Gram-Schmidt at loss computation time)

    Architecture:
    - Frozen encoder extracts per-frame features
    - Sequential core processes temporal sequence (parallel for training)
    - Per-frame: pool spatial tokens -> MLP head -> 9D pose delta

    Output: (B, T, 9) — T frame-to-frame deltas (frame 0 output is
    the delta from an implicit "pre-frame" and is typically masked in loss).
    """

    POSE_DIM = 9  # 3 translation + 6 rotation

    def __init__(
        self,
        encoder: nn.Module,
        d_enc: int,
        sequential_core: nn.Module,
        d_seq_out: int,
        freeze_encoder: bool = True,
        readout_d_model: int = 1024,
        pose_mlp_hidden_dim: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        self.d_enc = d_enc

        self.sequential_core = sequential_core

        self.proj_to_readout = nn.Linear(d_enc, readout_d_model) if d_enc != readout_d_model else nn.Identity()

        # Per-frame pose head: pool spatial tokens then predict 9D delta
        self.pose_head = nn.Sequential(
            nn.LayerNorm(readout_d_model),
            nn.Linear(readout_d_model, pose_mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(pose_mlp_hidden_dim, self.POSE_DIM),
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                for p in self.encoder.interm_mlps.parameters():
                    p.requires_grad_(True)
                self.encoder.interm_mlps.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
            if hasattr(self.encoder, 'interm_mlps'):
                self.encoder.interm_mlps.train(mode)
        return self

    def get_train_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def load_encoder_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into the encoder only."""
        print(f"Loading encoder weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        encoder_state = {}
        for k, v in ckpt.items():
            if any(k.startswith(prefix) for prefix in ['rvm_core.', 'head.', 'sequential_core.', 'pooler.', 'readout.', 'bbox_', 'pose_']):
                continue
            encoder_state[k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in encoder_state]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(encoder_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.encoder.load_state_dict(encoder_state, strict=strict)

        encoder_keys = set(self.encoder.state_dict().keys())
        loaded_keys = encoder_keys - set(m)
        print(f"  Encoder: {len(encoder_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (encoder keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in encoder): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def load_encoder_and_core_weights(self, checkpoint_path: str, strict: bool = False):
        """Load weights into both encoder and sequential_core."""
        print(f"Loading encoder + sequential_core weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

        new_state = {}
        for k, v in ckpt.items():
            if k.startswith('rvm_core.'):
                new_state['sequential_core.' + k[len('rvm_core.'):]] = v
            elif k.startswith('encoder.'):
                new_state['encoder.' + k] = v
            elif k in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
                new_state['encoder.' + k] = v

        skipped_ckpt_keys = [k for k in ckpt if k not in new_state
                             and not any(k.startswith(p) for p in ['rvm_core.', 'encoder.'])
                             and k not in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']]
        print(f"  Checkpoint: {len(ckpt)} keys total, {len(new_state)} mapped, {len(skipped_ckpt_keys)} skipped")
        if skipped_ckpt_keys:
            print(f"  Skipped checkpoint keys: {skipped_ckpt_keys[:10]}{'...' if len(skipped_ckpt_keys) > 10 else ''}")

        m, u = self.load_state_dict(new_state, strict=strict)

        model_keys = set(self.state_dict().keys())
        loaded_keys = model_keys - set(m)
        print(f"  Model: {len(model_keys)} keys total, {len(loaded_keys)} loaded, {len(m)} missing, {len(u)} unexpected")
        if m:
            print(f"  Missing (model keys NOT loaded): {m[:10]}{'...' if len(m) > 10 else ''}")
        if u:
            print(f"  Unexpected (ckpt keys NOT in model): {u[:10]}{'...' if len(u) > 10 else ''}")
        return m, u

    def forward(
        self,
        frames: torch.Tensor,
        encoder_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) video frames
            encoder_batch_size: optional chunked encoding

        Returns:
            pose_deltas: (B, T, 9) per-frame pose deltas
                [:, :, :3] = translation (dx, dy, dz)
                [:, :, 3:] = 6D rotation
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        # 1. Encode all frames
        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                tokens = self.encoder(frames_flat)
        else:
            tokens_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if (self.freeze_encoder and not getattr(self.encoder, 'mix_interm_feats', False)) else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                tokens_chunks.append(tokens_chunk)
            tokens = torch.cat(tokens_chunks, dim=0)

        if tokens.dim() == 2:
            tokens = tokens.view(B, T, -1)
        elif tokens.dim() == 3:
            N, D = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(B, T, N, D)

        # 2. Sequential core
        if hasattr(self.sequential_core, 'forward_sequence'):
            tokens = self.sequential_core.forward_sequence(tokens)
        else:
            tokens = self.sequential_core(tokens, return_sequence=True)

        # 3. Project to readout dim
        tokens = self.proj_to_readout(tokens)

        # 4. Pool spatial tokens per frame and predict pose
        if tokens.dim() == 4:
            # (B, T, N, D) -> pool over N -> (B, T, D)
            pooled = tokens.mean(dim=2)
        else:
            pooled = tokens  # already (B, T, D)

        pose_deltas = self.pose_head(pooled)  # (B, T, 9)
        return pose_deltas


