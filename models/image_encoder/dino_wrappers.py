"""DINO encoder wrappers for use with sequential models.

This module provides wrapper classes that adapt DINOv2 and DINOv3 output format
to be compatible with the sequential wrappers (ImageSequentialWrapper1F, etc.).
"""

import os
import torch
import torch.nn as nn
from typing import List, Literal, Optional


DINOOutputMode = Literal["patch_tokens", "cls_token", "cls_and_patch_mean", "all_tokens"]
DINOIntermMLPType = Literal["v1"]  # legacy alias for "mlp"
DINOIntermAggregator = Literal["mean", "xattn"]
DINOIntermProcessor = Literal[
    "mlp",            # MLP residual at every layer (current default)
]
DINOIntermLayers = Literal[
    # New canonical names
    "4_regular",            # 4 evenly spaced layers
    "3_regular",            # 3 evenly spaced layers
    "3_regular_and_conv",   # raw conv + 3 evenly spaced
    "4_regular_and_conv",   # raw conv + 4 evenly spaced
    "with_layer_0",         # layer 0 + 3 evenly spaced
    # Backward-compat aliases (deprecated)
    "standard",             # alias for 4_regular
    "with_embed",           # alias for 3_regular_and_conv
]

# Path to the local DINOv3 hub directory for torch.hub.load
DINOV3_HUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dinov3"))



def build_dinov3_encoder(
    model_name: str = "dinov3_vitb16",
    output_mode: DINOOutputMode = "patch_tokens",
    pretrained_path: Optional[str] = None,
    freeze: bool = True,
    mix_interm_feats: bool = False,
    interm_mlp_type: DINOIntermMLPType = "v1",
    interm_layers: DINOIntermLayers = "standard",
    interm_aggregator: DINOIntermAggregator = "mean",
    interm_processor: DINOIntermProcessor = "mlp",
    seq_len: Optional[int] = None,
) -> "DINOEncoder":
    """
    Build a DINOv3 encoder wrapped for use with sequential models.

    Args:
        model_name: Model variant, e.g. "dinov3_vitb16", "dinov3_vits16", "dinov3_vitl16"
        output_mode: How to extract features:
            - "patch_tokens": (B, N, D) patch tokens
            - "cls_token": (B, D) CLS token only
            - "cls_and_patch_mean": (B, 2*D) CLS + mean(patches)
            - "all_tokens": (B, 1+S+N, D) CLS + storage/registers + patches
        pretrained_path: Path to pretrained checkpoint, or None to download from URL
        freeze: Whether to freeze encoder weights
        mix_interm_feats: Whether to aggregate features from intermediate layers

    Returns:
        DINOEncoder instance ready for use with ImageSequentialWrapper1F
    """
    # DINOv3 uses torch.hub.load with local source
    # If pretrained_path is provided, pass it as the 'weights' argument
    if pretrained_path:
        print(f"Loading DINOv3 ({model_name}) from local checkpoint: {pretrained_path}...")
        backbone = torch.hub.load(
            DINOV3_HUB_DIR,
            model_name,
            source='local',
            weights=pretrained_path,
        )
    else:
        print(f"Loading DINOv3 ({model_name}) from default URL...")
        backbone = torch.hub.load(
            DINOV3_HUB_DIR,
            model_name,
            source='local',
            pretrained=True,
        )

    # Wrap with our adapter
    encoder = DINOEncoder(
        encoder=backbone,
        output_mode=output_mode,
        freeze=freeze,
        mix_interm_feats=mix_interm_feats,
        interm_mlp_type=interm_mlp_type,
        interm_layers=interm_layers,
        interm_aggregator=interm_aggregator,
        interm_processor=interm_processor,
        seq_len=seq_len,
    )

    return encoder


class _IntermFeatMLP(nn.Module):
    """Per-layer enrichment: Ti = Fi + MLP(BN(Fi)), expansion factor 2.

    Fi is already normalized by the model's frozen LN (via get_intermediate_layers(norm=True)),
    so the full formula is Ti = Fi + MLP(BN(LN_frozen(Fi_raw))).
    BN is BatchNorm1d applied over the (B*N, D) token dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) — already LN-normalized
        B, N, D = x.shape
        normed = self.bn(x.reshape(B * N, D)).reshape(B, N, D)
        return x + self.mlp(normed)


class _IntermFeatCrossAttnAggregator(nn.Module):
    """Per-position cross-attention aggregator over K intermediate layers.

    For each spatial position independently, treats the K layer features as a
    short sequence and applies cross-attention. The highest layer (most semantic)
    serves as the query, attending over all K layers (including itself).

    Fast because K is small (typically 4): attention cost is O(B*N * K^2 * D).
    With K=4, the K^2=16 cost is negligible.

    Architecture:
        1. Add learnable per-layer embeddings to KV (so the model can distinguish layers)
        2. Cross-attention: q=highest_layer, kv=all_layers
        3. Residual + MLP (standard transformer block)

    Args:
        dim: Feature dimension D.
        num_layers: Number of intermediate layers K to aggregate.
        num_heads: Attention heads (default 8).
        mlp_ratio: MLP expansion ratio (default 4).
    """
    def __init__(self, dim: int, num_layers: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        # Per-layer embeddings (let model distinguish features by source layer)
        self.layer_embed = nn.Parameter(torch.randn(num_layers, dim) * 0.02)
        # Cross-attention block (pre-norm)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True,
        )
        # MLP with residual
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, feat_list):
        """
        Args:
            feat_list: list of K tensors, each (B, N, D)
        Returns:
            (B, N, D) aggregated features
        """
        assert len(feat_list) == self.num_layers, \
            f"Expected {self.num_layers} layers, got {len(feat_list)}"
        B, N, D = feat_list[0].shape
        K = self.num_layers

        # Stack layers: list of K (B, N, D) → (B, N, K, D)
        # .contiguous() ensures layout is contiguous before reshape (avoids
        # non-contiguous strides that confuse flash attention kernels).
        kv_layers = torch.stack(feat_list, dim=2).contiguous()  # (B, N, K, D)
        # Add per-layer embedding (broadcast over B, N)
        kv_layers = kv_layers + self.layer_embed.view(1, 1, K, D)

        # Reshape to per-position sequences: (B*N, K, D)
        kv = kv_layers.reshape(B * N, K, D).contiguous()

        # Query = highest layer feature at each position: (B*N, 1, D)
        q = feat_list[-1].reshape(B * N, 1, D).contiguous()

        # Cross-attention with pre-norm
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        # Force the math (non-flash) SDPA kernel: K is tiny (~4), and flash
        # attention has constraints (alignment, head_dim limits, very large
        # batch dim) that can fail with this odd shape. The math kernel is
        # essentially free here since we only have ~4 KV tokens per position.
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.MATH
        ):
            attn_out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        x = q + attn_out  # residual on query

        # MLP with residual
        x = x + self.mlp(self.norm_mlp(x))

        return x.reshape(B, N, D)



class DINOEncoder(nn.Module):
    """
    Wrapper around DINOv2/DINOv3's DinoVisionTransformer that outputs patch tokens (B, N, D)
    in the same format as VideoMAE, so it can be used with VideoSequentialWrapper2F
    (with encoder_clip_len=1) or ImageSequentialWrapper1F.

    Args:
        encoder: DinoVisionTransformer instance (from DINOv2 or DINOv3)
        output_mode: how to extract features from DINO's dict output
            - "patch_tokens": return normalized patch tokens (B, N, D)
            - "cls_token": return CLS token only (B, D)
            - "cls_and_patch_mean": concatenate CLS with mean of patch tokens (B, 2*D)
            - "all_tokens": concatenate CLS + storage/registers + patches (B, 1+S+N, D)
        freeze: whether to freeze encoder weights
        mix_interm_feats: if True, aggregate patch features from K=4 equally-spaced
            intermediate layers [depth//4, depth//2, 3*depth//4, depth].
            For "all_tokens", only patch tokens are aggregated; CLS and storage tokens
            are taken from the final layer.
        interm_mlp_type: which per-layer enrichment block to use (only when mix_interm_feats=True):
            - "v1": Ti = Fi + MLP(LN(Fi))
            - "v2": Ti = Fi + MLP(BN(Fi))  where Fi is already LN-normalized by the frozen model norm
    """
    def __init__(
        self,
        encoder: nn.Module,
        output_mode: DINOOutputMode = "patch_tokens",
        freeze: bool = True,
        mix_interm_feats: bool = False,
        interm_mlp_type: DINOIntermMLPType = "v1",
        interm_layers: DINOIntermLayers = "standard",
        interm_aggregator: DINOIntermAggregator = "mean",
        interm_processor: DINOIntermProcessor = "mlp",
        seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_mode = output_mode
        self.embed_dim = encoder.embed_dim
        self.mix_interm_feats = mix_interm_feats

        # Number of storage/register tokens (DINOv3: 4, DINOv2-L w/ reg: 4, DINOv2-B: 0)
        self.num_storage_tokens = getattr(encoder, "n_storage_tokens", 0) or getattr(encoder, "num_register_tokens", 0)

        # Output dimension depends on mode
        if output_mode == "cls_and_patch_mean":
            self.output_dim = 2 * self.embed_dim
        else:
            self.output_dim = self.embed_dim

        # Intermediate feature aggregation
        if mix_interm_feats:
            depth = encoder.n_blocks
            # Backward-compat aliases
            interm_layers_resolved = interm_layers
            if interm_layers == "standard":
                interm_layers_resolved = "4_regular"
            elif interm_layers == "with_embed":
                interm_layers_resolved = "3_regular_and_conv"

            # 0-based layer indices (get_intermediate_layers uses 0-based block loop index)
            if interm_layers_resolved == "4_regular":
                # 4 evenly spaced layers: Large [5,11,17,23], Base [2,5,8,11]
                self._interm_layer_indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
            elif interm_layers_resolved == "3_regular":
                # 3 evenly spaced layers: Large [7,15,23], Base [3,7,11]
                self._interm_layer_indices = [depth // 3 - 1, 2 * depth // 3 - 1, depth - 1]
            elif interm_layers_resolved == "4_regular_and_conv":
                # Raw conv (-1) + 4 evenly spaced: Large [-1,5,11,17,23], Base [-1,2,5,8,11]
                self._interm_layer_indices = [-1, depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
            elif interm_layers_resolved == "3_regular_and_conv":
                # Raw conv (-1) + 3 evenly spaced: Large [-1,7,15,23], Base [-1,3,7,11]
                self._interm_layer_indices = [-1, depth // 3 - 1, 2 * depth // 3 - 1, depth - 1]
            elif interm_layers_resolved == "with_layer_0":
                # Layer 0 + 3 evenly spaced: Large [0,7,15,23], Base [0,3,7,11]
                self._interm_layer_indices = [0, depth // 3 - 1, 2 * depth // 3 - 1, depth - 1]
            else:
                raise ValueError(f"Unknown interm_layers mode: {interm_layers!r}")

            # ============ Per-layer enrichment (interm_processor) ============
            self.interm_mlp_type = interm_mlp_type  # legacy field, kept for compat
            self.interm_processor = interm_processor

            if interm_processor == "mlp":
                # Plain MLP residual at every layer
                self.interm_mlps = nn.ModuleList([
                    _IntermFeatMLP(self.embed_dim) for _ in range(len(self._interm_layer_indices))
                ])
            else:
                raise ValueError(f"Unknown interm_processor: {interm_processor}")

            # ============ Aggregation (interm_aggregator) ============
            self.interm_aggregator_type = interm_aggregator
            if interm_aggregator == "mean":
                self.interm_aggregator = None  # mean is computed inline in forward
            elif interm_aggregator == "xattn":
                self.interm_aggregator = _IntermFeatCrossAttnAggregator(
                    dim=self.embed_dim,
                    num_layers=len(self._interm_layer_indices),
                    num_heads=8,
                    mlp_ratio=4,
                )
            else:
                raise ValueError(f"Unknown interm_aggregator: {interm_aggregator}")

            print(f"  mix_interm_feats: layers {self._interm_layer_indices} (depth={depth}), "
                  f"mode={interm_layers_resolved}, processor={interm_processor}, "
                  f"aggregator={interm_aggregator}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        self._frozen = freeze

    def train(self, mode: bool = True):
        super().train(mode)
        if self._frozen:
            self.encoder.eval()
        return self

    def _get_interm_layer_feats(self, x: torch.Tensor):
        """
        Extract features from intermediate layers via get_intermediate_layers.
        Returns (patch_tokens_list, cls_tokens_list, extra_tokens_list) where
        each list has K entries. cls and extra tokens come from the raw sequence
        at position 0 and 1..S+1 respectively (already norm-ed).
        Uses return_class_token + return_extra_tokens if available (DINOv3),
        otherwise return_class_token only (DINOv2, no storage tokens).

        Special index -1 means raw patch embeddings (after patch_embed + pos_encoding,
        before any transformer blocks).
        """
        # Separate -1 (embed) from real layer indices
        has_embed = -1 in self._interm_layer_indices
        if has_embed:
            real_indices = [i for i in self._interm_layer_indices if i != -1]
        else:
            real_indices = self._interm_layer_indices

        # Get intermediate layer features (only real indices)
        try:
            # DINOv3: supports return_extra_tokens
            layer_feats = self.encoder.get_intermediate_layers(
                x, n=real_indices, norm=True,
                return_class_token=True, return_extra_tokens=True,
            )  # tuple of K tuples: (patches, cls, extra)
            patch_list  = [f[0] for f in layer_feats]  # each (B, N, D)
            cls_list    = [f[1] for f in layer_feats]  # each (B, D)
            extra_list  = [f[2] for f in layer_feats]  # each (B, S, D)
        except TypeError:
            # Fallback for encoders that don't support return_extra_tokens
            layer_feats = self.encoder.get_intermediate_layers(
                x, n=real_indices, norm=True,
                return_class_token=True,
            )  # tuple of K tuples: (patches, cls)
            patch_list  = [f[0] for f in layer_feats]
            cls_list    = [f[1] for f in layer_feats]
            extra_list  = [torch.zeros(f[0].shape[0], 0, f[0].shape[2], device=x.device)
                           for f in layer_feats]

        # Insert raw embed tokens at the position where -1 appears
        if has_embed:
            S = self.num_storage_tokens
            embed = self.encoder.prepare_tokens_with_masks(x)
            # DINOv3 returns (tensor, (H, W)); DINOv2 returns just tensor
            if isinstance(embed, tuple):
                embed = embed[0]  # (B, 1+S+N, D)
            embed_patches = embed[:, 1 + S:]     # (B, N, D)
            embed_cls = embed[:, 0]              # (B, D)
            embed_extra = embed[:, 1:1 + S]      # (B, S, D)

            # Find insertion position of -1 in the original list
            insert_idx = self._interm_layer_indices.index(-1)
            patch_list.insert(insert_idx, embed_patches)
            cls_list.insert(insert_idx, embed_cls)
            extra_list.insert(insert_idx, embed_extra)

        return patch_list, cls_list, extra_list

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) single images or (B*T, C, H, W) flattened temporal batch
            seq_len: optional T hint, unused by the shipped processors (kept for API parity).
        Returns:
            features: (B, N, D) patch tokens, (B, D) CLS token, (B, 2*D) concatenated,
                      or (B, 1+S+N, D) all tokens (CLS + storage/registers + patches)
        """
        if self.mix_interm_feats:
            # Run frozen backbone under no_grad (memory efficiency);
            # trainable interm_mlps run outside no_grad to allow gradient flow
            if self._frozen:
                with torch.no_grad():
                    patch_list, cls_list, extra_list = self._get_interm_layer_feats(x)
            else:
                patch_list, cls_list, extra_list = self._get_interm_layer_feats(x)

            # Step 1: per-layer enrichment (Ti = block(Fi))
            # Pass seq_len hint so temporal processors (GMM) can adapt to the
            # actual batch structure.
            enriched = []
            for f, mlp in zip(patch_list, self.interm_mlps):
                # Temporal processors accept a `seq_len` kwarg; plain MLPs ignore it
                try:
                    enriched.append(mlp(f, seq_len=seq_len))
                except TypeError:
                    enriched.append(mlp(f))

            # Step 2: aggregate K enriched layers into one feature map
            if self.interm_aggregator is not None:
                patch_tokens = self.interm_aggregator(enriched)  # (B, N, D)
            else:
                # Mean aggregation
                patch_tokens = sum(enriched) / len(enriched)  # (B, N, D)

            # CLS and storage from the final layer (last entry)
            cls_token    = cls_list[-1]    # (B, D)
            extra_tokens = extra_list[-1]  # (B, S, D)

            if self.output_mode == "patch_tokens":
                return patch_tokens
            elif self.output_mode == "cls_token":
                return cls_token
            elif self.output_mode == "cls_and_patch_mean":
                return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
            elif self.output_mode == "all_tokens":
                return torch.cat([cls_token.unsqueeze(1), extra_tokens, patch_tokens], dim=1)
            else:
                raise ValueError(f"Unknown output_mode: {self.output_mode}")
        else:
            # Standard: use forward_features output directly
            out = self.encoder.forward_features(x)

            if self.output_mode == "patch_tokens":
                return out["x_norm_patchtokens"]  # (B, N, D)
            elif self.output_mode == "cls_token":
                return out["x_norm_clstoken"]  # (B, D)
            elif self.output_mode == "cls_and_patch_mean":
                cls_token = out["x_norm_clstoken"]           # (B, D)
                patch_mean = out["x_norm_patchtokens"].mean(dim=1)  # (B, D)
                return torch.cat([cls_token, patch_mean], dim=-1)  # (B, 2*D)
            elif self.output_mode == "all_tokens":
                # Concatenate CLS + storage/register tokens + patch tokens along token dim
                # DINOv2: [CLS | patches] (no storage tokens)
                # DINOv3: [CLS | storage_0..3 | patches]
                cls_token = out["x_norm_clstoken"].unsqueeze(1)  # (B, 1, D)
                patch_tokens = out["x_norm_patchtokens"]          # (B, N, D)
                # DINOv3 exposes storage tokens as "x_storage_tokens"; DINOv2 uses "x_norm_regtokens"
                storage_key = "x_storage_tokens" if "x_storage_tokens" in out else "x_norm_regtokens"
                storage_tokens = out[storage_key]  # (B, S, D)
                return torch.cat([cls_token, storage_tokens, patch_tokens], dim=1)  # (B, 1+S+N, D)
            else:
                raise ValueError(f"Unknown output_mode: {self.output_mode}")
