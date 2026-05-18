"""Mamba-based recurrent temporal cores used in the paper.

Implements the three temporal architectures from Table 1, applied per-patch
across time:

- MambaSeqCore           ("M"     in the paper)
- MambaMixSeqCore        ("MMix"  in the paper)
- GatedMambaMixSeqCore   ("GMMix" in the paper)

Each core exposes the same recurrent interface: a stateless `forward` over
(B, T, N, D) tensors and a `step` / `init_state` / `reset_state` interface for
streaming inference using mamba_ssm's `InferenceParams`.

The streaming path treats each spatial token as an independent time series:
B * N parallel Mamba sequences, sharing the same model weights.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

from models.rvm_modules.rvm_blocks import TransformerMLP


# ---------------------------------------------------------------------------
# Stochastic Depth (DropPath)
# ---------------------------------------------------------------------------

def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic Depth per sample, applied on residual paths."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class MambaResidualBlockStateful(nn.Module):
    """y = x + Mamba(LayerNorm(x))  (stateful: supports streaming via InferenceParams)."""

    def __init__(self, d_model=768, d_state=16, expand=2, dropout=0.1, layer_idx=None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=expand, layer_idx=layer_idx)
        self.dropout = nn.Dropout(dropout)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        u = self.norm(x)
        v = self.mamba(u)  # stateless
        return x + self.dropout(v)

    def forward_step(self, x_t: torch.Tensor, ip: InferenceParams) -> torch.Tensor:
        # x_t: (B, D)
        x1 = x_t.unsqueeze(1)                       # (B, 1, D)
        u = self.norm(x1)
        v = self.mamba(u, inference_params=ip)      # stateful
        y = x1 + self.dropout(v)
        return y.squeeze(1)                         # (B, D)


class SpatialSelfAttentionBlock(nn.Module):
    """Per-frame spatial self-attention + MLP with DropPath on residuals:
        y = x + DropPath(SelfAttn(LayerNorm(x))) + DropPath(MLP(LayerNorm(x)))
    """

    def __init__(self, d_model: int, num_heads: int = 12, mlp_ratio: int = 4,
                 dropout: float = 0.0, drop_path: float = 0.1):
        super().__init__()
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, d_model * mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, N, D) -> (B*T, N, D)"""
        x_norm = self.sa_norm(x)
        sa_out, _ = self.sa_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path(sa_out)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))
        return x


# ---------------------------------------------------------------------------
# "M" — pure temporal Mamba per patch
# ---------------------------------------------------------------------------

class MambaSeqCore(nn.Module):
    """Pure temporal Mamba per patch (paper "M"):
        y_t = MambaStep(x_t)

    Batched (train/val/test):
        x: (B, T, N, D)  ->  (B, T, N, D)

    Streaming:
        step(x_t, state):  (B, N, D)  ->  (B, N, D), new_state

    Uses chunked processing to bound memory with large (B * N).
    """
    _supports_parallel = True

    def __init__(
        self,
        d_model: int = 768,
        n_mamba_layers: int = 1,
        d_state: int = 16,
        expand: int = 2,
        mamba_dropout: float = 0.1,
        max_seqlen: int = 4096,
        num_patches: int = 196,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seqlen = max_seqlen
        self.num_patches = num_patches
        self.chunk_size = chunk_size

        self.mamba_blocks = nn.ModuleList([
            MambaResidualBlockStateful(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dropout=mamba_dropout,
                layer_idx=i,
            )
            for i in range(n_mamba_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

    def _mamba_forward_chunked(self, z: torch.Tensor) -> torch.Tensor:
        """z: (total_seqs, T, D) -> (total_seqs, T, D); chunked along batch dim."""
        total_seqs = z.shape[0]

        if self.chunk_size <= 0 or total_seqs <= self.chunk_size:
            for blk in self.mamba_blocks:
                z = blk.forward_full(z)
            return z

        chunks_out = []
        for start in range(0, total_seqs, self.chunk_size):
            end = min(start + self.chunk_size, total_seqs)
            chunk = z[start:end].contiguous()
            for blk in self.mamba_blocks:
                chunk = blk.forward_full(chunk)
            chunks_out.append(chunk)

        return torch.cat(chunks_out, dim=0)

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)                                # (B, N, D) -> (B, 1, N, D)

        B, T, N, D = x.shape

        z = x.permute(0, 2, 1, 3).reshape(B * N, T, D)       # (B*N, T, D)
        z = self._mamba_forward_chunked(z)                    # (B*N, T, D)
        z = z.reshape(B, N, T, D).permute(0, 2, 1, 3)         # (B, T, N, D)

        z = self.output_norm(z)
        if return_sequence:
            return z
        return z.mean(dim=(1, 2))                             # (B, D)

    def init_state(self, batch_size: int, device: torch.device) -> InferenceParams:
        effective_batch = batch_size * self.num_patches
        ip = InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=effective_batch)
        ip.reset(self.max_seqlen, effective_batch)
        return ip

    def reset_state(self, batch_size: int, device: torch.device, state: Optional[InferenceParams] = None) -> InferenceParams:
        if state is None:
            return self.init_state(batch_size, device)
        effective_batch = batch_size * self.num_patches
        state.reset(self.max_seqlen, effective_batch)
        return state

    def step(self, x_t: torch.Tensor, state: InferenceParams) -> Tuple[torch.Tensor, InferenceParams]:
        if x_t.dim() != 3:
            raise ValueError(f"Expected x_t (B,N,D), got {x_t.shape}")

        B, N, D = x_t.shape
        z_t = x_t.reshape(B * N, D)

        for blk in self.mamba_blocks:
            z_t = blk.forward_step(z_t, state)

        state.seqlen_offset += 1

        z_t = z_t.reshape(B, N, D)
        z_t = self.output_norm(z_t)
        return z_t, state


# ---------------------------------------------------------------------------
# "MMix" — interleaved spatial self-attention + temporal Mamba
# ---------------------------------------------------------------------------

class MambaMixSeqCore(nn.Module):
    """Spatial self-attn + temporal Mamba (paper "MMix"):
        for each layer:
          z = z + SelfAttn(z)     (per frame, across N patches)
          z = z + Mamba(z)        (per patch, across T frames)
    """
    _supports_parallel = True

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 1,
        d_state: int = 16,
        expand: int = 2,
        mamba_dropout: float = 0.1,
        spatial_num_heads: int = 12,
        spatial_mlp_ratio: int = 4,
        max_seqlen: int = 4096,
        num_patches: int = 196,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seqlen = max_seqlen
        self.num_patches = num_patches
        self.chunk_size = chunk_size
        self.n_layers = n_layers

        self.spatial_blocks = nn.ModuleList([
            SpatialSelfAttentionBlock(
                d_model=d_model,
                num_heads=spatial_num_heads,
                mlp_ratio=spatial_mlp_ratio,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])
        self.mamba_blocks = nn.ModuleList([
            MambaResidualBlockStateful(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dropout=mamba_dropout,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

    def _forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """One spatial-then-temporal layer. x: (B, T, N, D) -> (B, T, N, D)"""
        B, T, N, D = x.shape

        x_spatial = x.reshape(B * T, N, D)
        x_spatial = self.spatial_blocks[layer_idx](x_spatial)

        x_temporal = x_spatial.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)

        mamba_block = self.mamba_blocks[layer_idx]
        total_seqs = x_temporal.shape[0]
        if self.chunk_size <= 0 or total_seqs <= self.chunk_size:
            x_temporal = mamba_block.forward_full(x_temporal)
        else:
            chunks_out = []
            for start in range(0, total_seqs, self.chunk_size):
                end = min(start + self.chunk_size, total_seqs)
                chunk = x_temporal[start:end].contiguous()
                chunk = mamba_block.forward_full(chunk)
                chunks_out.append(chunk)
            x_temporal = torch.cat(chunks_out, dim=0)

        return x_temporal.reshape(B, N, T, D).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B, T, N, D = x.shape

        z = x
        for i in range(self.n_layers):
            z = self._forward_layer(z, i)

        z = self.output_norm(z)
        if return_sequence:
            return z
        return z.mean(dim=(1, 2))

    def init_state(self, batch_size: int, device: torch.device) -> InferenceParams:
        effective_batch = batch_size * self.num_patches
        ip = InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=effective_batch)
        ip.reset(self.max_seqlen, effective_batch)
        return ip

    def reset_state(self, batch_size: int, device: torch.device, state: Optional[InferenceParams] = None) -> InferenceParams:
        if state is None:
            return self.init_state(batch_size, device)
        effective_batch = batch_size * self.num_patches
        state.reset(self.max_seqlen, effective_batch)
        return state

    def step(self, x_t: torch.Tensor, state: InferenceParams) -> Tuple[torch.Tensor, InferenceParams]:
        if x_t.dim() != 3:
            raise ValueError(f"Expected x_t (B,N,D), got {x_t.shape}")

        B, N, D = x_t.shape

        z_t = x_t
        for i in range(self.n_layers):
            z_t = self.spatial_blocks[i](z_t)
            z_flat = z_t.reshape(B * N, D)
            z_flat = self.mamba_blocks[i].forward_step(z_flat, state)
            z_t = z_flat.reshape(B, N, D)

        state.seqlen_offset += 1

        z_t = self.output_norm(z_t)
        return z_t, state


# ---------------------------------------------------------------------------
# "GMMix" — MMix + learned gate that blends pre- vs post-Mamba representations
# ---------------------------------------------------------------------------

class GatedMambaMixSeqCore(nn.Module):
    """Spatial self-attn + gated temporal Mamba (paper "GMMix"):
        for each layer:
          z = z + SelfAttn(z)            (per frame, across N patches)
          h = Mamba(z)                   (per patch, across T frames)
          u = sigmoid(W([z; h]))         (per token)
          z = (1 - u) * z + u * h
    """
    _supports_parallel = True

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 1,
        d_state: int = 16,
        expand: int = 2,
        mamba_dropout: float = 0.1,
        spatial_num_heads: int = 12,
        spatial_mlp_ratio: int = 4,
        gate_hidden: int = 0,
        max_seqlen: int = 4096,
        num_patches: int = 196,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seqlen = max_seqlen
        self.num_patches = num_patches
        self.chunk_size = chunk_size
        self.n_layers = n_layers

        self.spatial_blocks = nn.ModuleList([
            SpatialSelfAttentionBlock(
                d_model=d_model,
                num_heads=spatial_num_heads,
                mlp_ratio=spatial_mlp_ratio,
                dropout=0.1,
            )
            for _ in range(n_layers)
        ])
        self.mamba_blocks = nn.ModuleList([
            MambaResidualBlockStateful(d_model=d_model, d_state=d_state, expand=expand, dropout=mamba_dropout, layer_idx=i)
            for i in range(n_layers)
        ])

        if gate_hidden > 0:
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * d_model, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, d_model),
                )
                for _ in range(n_layers)
            ])
        else:
            self.gates = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_layers)])

        self.output_norm = nn.LayerNorm(d_model)

    def _mamba_forward_chunked(self, z: torch.Tensor, blk: nn.Module) -> torch.Tensor:
        total_seqs = z.shape[0]
        if self.chunk_size <= 0 or total_seqs <= self.chunk_size:
            return blk.forward_full(z)
        outs = []
        for start in range(0, total_seqs, self.chunk_size):
            end = min(start + self.chunk_size, total_seqs)
            outs.append(blk.forward_full(z[start:end].contiguous()))
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, T, N, D = x.shape
        if N != self.num_patches:
            raise ValueError(f"Expected N={self.num_patches}, got {N}")

        z = x
        for i in range(self.n_layers):
            z_sp = self.spatial_blocks[i](z.reshape(B * T, N, D)).reshape(B, T, N, D)

            z_tmp = z_sp.permute(0, 2, 1, 3).reshape(B * N, T, D)   # (B*N, T, D)
            h_tmp = self._mamba_forward_chunked(z_tmp, self.mamba_blocks[i])

            u = torch.sigmoid(self.gates[i](torch.cat([z_tmp, h_tmp], dim=-1)))
            y_tmp = (1.0 - u) * z_tmp + u * h_tmp

            z = y_tmp.reshape(B, N, T, D).permute(0, 2, 1, 3)

        z = self.output_norm(z)
        if return_sequence:
            return z
        return z.mean(dim=(1, 2))

    def init_state(self, batch_size: int, device: torch.device) -> InferenceParams:
        effective_batch = batch_size * self.num_patches
        ip = InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=effective_batch)
        ip.reset(self.max_seqlen, effective_batch)
        return ip

    def reset_state(self, batch_size: int, device: torch.device, state: Optional[InferenceParams] = None) -> InferenceParams:
        if state is None:
            return self.init_state(batch_size, device)
        effective_batch = batch_size * self.num_patches
        state.reset(self.max_seqlen, effective_batch)
        return state

    def step(self, x_t: torch.Tensor, state: InferenceParams) -> Tuple[torch.Tensor, InferenceParams]:
        if x_t.dim() != 3:
            raise ValueError(f"Expected x_t (B,N,D), got {x_t.shape}")
        B, N, D = x_t.shape
        if N != self.num_patches:
            raise ValueError(f"Expected N={self.num_patches}, got {N}")

        z = x_t
        for i in range(self.n_layers):
            z_sp = self.spatial_blocks[i](z)                                 # (B, N, D)
            z_flat = z_sp.reshape(B * N, D)                                  # (B*N, D)
            h_flat = self.mamba_blocks[i].forward_step(z_flat, state)
            u = torch.sigmoid(self.gates[i](torch.cat([z_flat, h_flat], dim=-1)))
            y_flat = (1.0 - u) * z_flat + u * h_flat
            z = y_flat.reshape(B, N, D)

        state.seqlen_offset += 1

        z = self.output_norm(z)
        return z, state
