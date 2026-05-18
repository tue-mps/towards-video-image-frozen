"""RVM-style blocks for sequential video processing.

Pure PyTorch implementation of RVM (Recurrent Video Masked Autoencoders).

Components:
- TransformerMLP: standard Transformer MLP block with GELU.
- RVMReadout: 4DS-style attention readout (with optional V-JEPA matching).
- RVMCrossAttentionBlock / RVMCrossAttentionTransformer: cross-attention
  building blocks used inside the recurrent core.
- IdentityCore: pass-through recurrent core (useful for ablations).
- GatedTransformerCore: RVM's main recurrent module with GRU-style gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerMLP(nn.Module):
    """Standard Transformer MLP block with GELU activation."""

    def __init__(self, d_model: int, hidden_size: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_size = hidden_size or 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RVMReadout(nn.Module):
    """Precise PyTorch implementation of RVM's AttentionReadout.

    Follows the 4DS protocol as stated in the RVM paper:
    https://github.com/google-deepmind/representations4d/blob/main/representations4d/models/readout.py

    With match_vjepa_implementation=True (default, matches V-JEPA protocol):
    1. Input LayerNorm (on input_dim)
    2. Learned temporal positional embedding (added along time axis, on input_dim)
    3. Learnable query tokens (shape: num_queries x num_heads x head_dim)
    4. K/V projections from input_dim -> num_params WITH bias
    5. Single cross-attention: queries attend to features
    6. Query residual connection: output = query + Linear(attention_output)
    7. MLP block with residual AROUND LayerNorm+MLP
    8. Final classification projection

    With match_vjepa_implementation=False (simpler version):
    1. NO input LayerNorm
    2. Learned temporal positional embedding (if add_temporal_posenc=True)
    3. Learnable query tokens
    4. K/V projections WITHOUT bias
    5. Single cross-attention
    6. NO query residual, NO MLP block
    7. Direct classification projection

    Input shape: (B, T, N, C) - batch, time, spatial tokens, channels (C = input_dim)
    Output shape: (B, num_queries, num_classes) or (B, num_classes) if num_queries=1
    """

    def __init__(
        self,
        num_params: int,
        num_classes: int,
        num_queries: int,
        num_heads: int = 16,
        num_frames: int = 16,
        mlp_ratio: int = 4,
        input_dim: Optional[int] = None,
        match_vjepa_implementation: bool = True,
        add_temporal_posenc: bool = True,
    ):
        """
        Args:
            num_params: Internal readout dimension (e.g., 768 or 1024)
            num_classes: Number of output classes
            num_queries: Number of learnable query tokens (Q)
            num_heads: Number of attention heads
            num_frames: Number of temporal frames for positional embedding
            mlp_ratio: MLP hidden dimension multiplier
            input_dim: Input feature dimension. If None, defaults to num_params.
                       Use this when encoder output dim differs from readout dim.
            match_vjepa_implementation: If True (default), adds input LayerNorm,
                       K/V bias, query residual, and MLP block (matching V-JEPA).
                       If False, uses simpler cross-attention only.
            add_temporal_posenc: If True (default), adds learned temporal positional
                       embedding before attention.
        """
        super().__init__()
        self.num_params = num_params
        self.input_dim = input_dim if input_dim is not None else num_params
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = num_params // num_heads
        self.match_vjepa_implementation = match_vjepa_implementation
        self.add_temporal_posenc = add_temporal_posenc

        assert num_params % num_heads == 0, f"num_params ({num_params}) must be divisible by num_heads ({num_heads})"

        # Input normalization (only with match_vjepa_implementation)
        if match_vjepa_implementation:
            self.input_norm = nn.LayerNorm(self.input_dim)
        else:
            self.input_norm = None

        # Learned temporal positional embedding: (1, T, 1, input_dim)
        # Initialized with normal(0.02) as in original (BERT-style)
        if add_temporal_posenc:
            self.temporal_posenc = nn.Parameter(torch.randn(1, num_frames, 1, self.input_dim) * 0.02)
        else:
            self.temporal_posenc = None

        # Learnable query tokens: (Q, num_heads, head_dim)
        # Original: self.param('query', normal(0.02), [num_queries, num_heads, head_dim])
        self.query = nn.Parameter(torch.randn(num_queries, num_heads, self.head_dim) * 0.02)

        # Key and Value projections: input_dim -> (num_heads * head_dim) = num_params
        # With match_vjepa_implementation: use_bias=True, otherwise use_bias=False
        use_bias = match_vjepa_implementation
        self.key_proj = nn.Linear(self.input_dim, num_params, bias=use_bias)
        self.value_proj = nn.Linear(self.input_dim, num_params, bias=use_bias)

        # Query residual and MLP block (only with match_vjepa_implementation)
        if match_vjepa_implementation:
            # Output projection after attention (for residual: token = query + Dense(token))
            self.out_proj = nn.Linear(num_params, num_params)
            # MLP block: LayerNorm + MLP with residual around
            self.mlp_norm = nn.LayerNorm(num_params)
            self.mlp_fc1 = nn.Linear(num_params, num_params * mlp_ratio)
            self.mlp_fc2 = nn.Linear(num_params * mlp_ratio, num_params)
        else:
            self.out_proj = None
            self.mlp_norm = None
            self.mlp_fc1 = None
            self.mlp_fc2 = None

        # Final classification head
        self.classifier = nn.Linear(num_params, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Features of shape (B, T, N, C) or (B, T*N, C)
                    B = batch, T = time, N = spatial tokens, C = input_dim

        Returns:
            Logits of shape (B, num_queries, num_classes) or (B, num_classes) if num_queries=1
        """
        # Handle both 4D (B, T, N, C) and 3D (B, T*N, C) inputs
        if inputs.dim() == 4:
            B, T, N, C = inputs.shape
            feats = inputs
        else:
            # If 3D, assume it's already flattened - no temporal posenc
            B, TN, C = inputs.shape
            feats = inputs
            T = None

        # 1. Input LayerNorm (only with match_vjepa_implementation)
        if self.input_norm is not None:
            feats = self.input_norm(feats)

        # 2. Add temporal positional embedding (only if 4D input and enabled)
        if T is not None and self.temporal_posenc is not None:
            # Interpolate temporal posenc if needed
            if T != self.temporal_posenc.shape[1]:
                posenc = F.interpolate(
                    self.temporal_posenc.permute(0, 3, 1, 2),  # (1, C, T_orig, 1)
                    size=(T, 1),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # (1, T, 1, C)
            else:
                posenc = self.temporal_posenc

            feats = feats + posenc  # (B, T, N, C)

        # 3. Flatten time and space: (B, T, N, C) -> (B, T*N, C)
        if T is not None:
            feats = feats.reshape(B, T * N, C)

        # 4. Project to key and value: (B, T*N, num_heads, head_dim)
        # K/V projection handles input_dim -> num_params dimension change
        seq_len = feats.shape[1]
        key = self.key_proj(feats).view(B, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(feats).view(B, seq_len, self.num_heads, self.head_dim)

        # 5. Expand query for batch: (B, Q, num_heads, head_dim)
        query = self.query.unsqueeze(0).expand(B, -1, -1, -1)

        # 6. Scaled dot-product attention
        # query: (B, Q, H, D), key: (B, S, H, D) -> attn: (B, H, Q, S)
        scale = self.head_dim ** -0.5
        attn_weights = torch.einsum('bqhd,bshd->bhqs', query * scale, key)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values: (B, H, Q, S) x (B, S, H, D) -> (B, Q, H, D)
        token = torch.einsum('bhqs,bshd->bqhd', attn_weights, value)

        # 7. Reshape: (B, Q, H, D) -> (B, Q, num_params)
        token = token.reshape(B, self.num_queries, self.num_params)

        # 8. Query residual and MLP (only with match_vjepa_implementation)
        if self.match_vjepa_implementation:
            query_flat = query.reshape(B, self.num_queries, self.num_params)

            # Query residual: token = query + Dense(token)
            token = query_flat + self.out_proj(token)

            # MLP block with residual AROUND LayerNorm+MLP
            residual = token
            token = self.mlp_norm(token)
            token = self.mlp_fc2(F.gelu(self.mlp_fc1(token)))
            token = token + residual

        # 9. Classification head
        out = self.classifier(token)  # (B, Q, num_classes)

        # 10. Squeeze if single query (matching original behavior)
        if self.num_queries == 1:
            out = out.squeeze(-2)  # (B, num_classes)

        return out


# =============================================================================
# RVM Original Architecture (GatedTransformerCore)
# From: https://github.com/google-deepmind/representations4d
# =============================================================================


class RVMCrossAttentionBlock(nn.Module):
    """Cross-attention block as used in RVM's CrossAttentionTransformer.

    Each block has three sub-layers in order:
    1. Cross-attention: queries attend to key-value pairs
    2. MLP: feed-forward network
    3. Self-attention: queries attend to themselves

    This matches the original RVM implementation exactly.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Cross-attention: query attends to key-value
        self.ca_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ca_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP
        self.mlp_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = TransformerMLP(d_model, d_model * mlp_ratio, dropout)

        # Self-attention
        self.sa_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.sa_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,     # (B, N, D) - input tokens (query)
        inputs_kv: torch.Tensor,  # (B, M, D) - key-value tokens
    ) -> torch.Tensor:
        """
        Args:
            inputs: Query tokens (B, N, D)
            inputs_kv: Key-value tokens (B, M, D)

        Returns:
            Updated tokens (B, N, D)
        """
        x = inputs

        # 1. Cross-attention: attend to key-value
        x_norm = self.ca_norm(x)
        ca_out, _ = self.ca_attn(x_norm, inputs_kv, inputs_kv, need_weights=False)
        x = x + ca_out

        # 2. MLP
        x = x + self.mlp(self.mlp_norm(x))

        # 3. Self-attention
        x_norm = self.sa_norm(x)
        sa_out, _ = self.sa_attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + sa_out

        return x


class RVMCrossAttentionTransformer(nn.Module):
    """Stack of RVM cross-attention blocks.

    This is the transformer component inside GatedTransformerCore.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RVMCrossAttentionBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        inputs: torch.Tensor,     # (B, N, D)
        inputs_kv: torch.Tensor,  # (B, M, D)
    ) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x, inputs_kv)
        return self.output_norm(x)


class IdentityCore(nn.Module):
    """No-op sequential core: passes tokens through unchanged.

    Use as a baseline to evaluate the encoder alone without any temporal processing.
    Implements the full sequential core interface: forward, forward_sequence,
    and streaming (init_state, reset_state, step).
    """

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_sequence: bool = True) -> torch.Tensor:
        x = self.output_norm(x)
        if return_sequence:
            return x
        return x.mean(dim=tuple(range(1, x.dim() - 1)))  # (B, D)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_norm(x)

    # -- streaming interface --
    def init_state(self, batch_size: int, *args, **kwargs):
        return None

    def reset_state(self, batch_size: int, *args, **kwargs):
        return None

    def step(self, x_t: torch.Tensor, state=None):
        """Single-frame step: (B, N, D) -> (B, N, D), state."""
        return self.output_norm(x_t), None


class GatedTransformerCore(nn.Module):
    """RVM's Gated Transformer Core (the original RVM recurrent core):
      for each frame t:
        u_t = sigmoid(Wu_x(x_t) + Wu_s(s_{t-1}))        (update gate)
        r_t = sigmoid(Wr_x(x_t) + Wr_s(s_{t-1}))        (reset gate)
        h_t = CrossAttnTransformer(Q=x_t, KV=r_t*LN(s_{t-1}))
        s_t = (1-u_t)*s_{t-1} + u_t*h_t

    PyTorch reimplementation of https://github.com/google-deepmind/representations4d
    """
    _supports_parallel = True

    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        num_patches: int = 196,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches

        # Gating mechanism (GRU-style, no bias as in original)
        self.input_update = nn.Linear(d_model, d_model, bias=False)
        self.input_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_update = nn.Linear(d_model, d_model, bias=False)
        self.state_reset = nn.Linear(d_model, d_model, bias=False)

        # State normalization (applied before gating, eps=1e-6 matching JAX default)
        self.state_layer_norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)

        # Cross-attention transformer
        self.transformer = RVMCrossAttentionTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

    def init_state(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Initialize state to zeros (as in original RVM)."""
        return torch.zeros(batch_size, num_tokens, self.d_model, device=device)

    def forward_step(
        self,
        inputs: torch.Tensor,  # (B, N, D) - current frame tokens
        state: torch.Tensor,   # (B, N, D) - previous hidden state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent step.

        Args:
            inputs: Current frame tokens (B, N, D)
            state: Previous hidden state (B, N, D)

        Returns:
            output: Output tokens (B, N, D)
            new_state: Updated state (B, N, D)
        """
        # Compute gates (element-wise on token dimension)
        update_gate = torch.sigmoid(
            self.input_update(inputs) + self.state_update(state)
        )  # (B, N, D)
        reset_gate = torch.sigmoid(
            self.input_reset(inputs) + self.state_reset(state)
        )  # (B, N, D)

        # Apply reset gate to normalized state
        gated_state = reset_gate * self.state_layer_norm(state)  # (B, N, D)

        # Transformer: input as query, gated_state as key-value
        h = self.transformer(inputs, gated_state)  # (B, N, D)

        # Blend with previous state using update gate
        output = (1 - update_gate) * state + update_gate * h  # (B, N, D)

        # State is same as output in RVM
        new_state = output

        return output, new_state

    def _core_is_frozen(self) -> bool:
        """Check if the core parameters (excluding output_norm) are frozen."""
        core_params = [p for n, p in self.named_parameters() if not n.startswith('output_norm')]
        return len(core_params) > 0 and all(not p.requires_grad for p in core_params)

    def _run_recurrent(
        self,
        x: torch.Tensor,  # (B, T, N, D)
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent core over a sequence. Returns (outputs, final_state)."""
        B, T, N, D = x.shape

        if state is None:
            state = self.init_state(B, N, x.device)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, D)
            out_t, state = self.forward_step(x_t, state)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)  # (B, T, N, D)
        return outputs, state

    def forward(
        self,
        x: torch.Tensor,  # (B, T, N, D) or (B, N, D)
        state: Optional[torch.Tensor] = None,
        return_sequence: Optional[bool] = None,
    ):
        """
        Process full video sequence recurrently.

        Args:
            x: Video tokens (B, T, N, D) or (B, N, D)
            state: Initial state (B, N, D) or None to initialize with zeros
            return_sequence: If True, return (B, T, N, D) tensor (for use as sequential_core).
                           If None (default), return (outputs, state) tuple (legacy interface).

        Returns:
            If return_sequence is True: (B, T, N, D) tensor
            If return_sequence is None: (outputs, state) tuple (legacy)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, N, D) -> (B, 1, N, D)

        # Auto-detect frozen core for memory savings
        core_frozen = self._core_is_frozen()
        if not hasattr(self, '_logged_frozen'):
            print(f"[GatedTransformerCore] core_frozen={core_frozen}, "
                  f"using {'torch.no_grad()' if core_frozen else 'torch.enable_grad()'}")
            self._logged_frozen = True
        ctx = torch.no_grad() if core_frozen else torch.enable_grad()

        with ctx:
            outputs, final_state = self._run_recurrent(x, state)

        outputs = self.output_norm(outputs)

        if return_sequence is not None:
            # New interface: return tensor directly
            if return_sequence:
                return outputs
            else:
                return outputs.mean(dim=(1, 2))  # (B, D)

        # Legacy interface: return tuple
        return outputs, final_state


