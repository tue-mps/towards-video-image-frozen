"""
Load RVM pretrained weights into our PyTorch implementation.

Maps the official JAX `.npz` RVM checkpoint to our ViT + GatedTransformerCore
PyTorch model. Both the Base and Large variants are supported via the
`MODEL_SIZE` toggle below.

Key mappings:
- JAX uses (input_dim, num_heads, head_dim) for attention, PyTorch uses (num_heads * head_dim, input_dim)
- JAX "kernel" = weight, "scale" = weight (for LayerNorm), "bias" = bias
- nn.MultiheadAttention uses fused in_proj_weight for Q,K,V

Reference: https://github.com/google-deepmind/representations4d/blob/main/colabs/rvm_inference_demo.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# =============================================================================
# Configuration
# =============================================================================

# Edit these two lines to convert a different checkpoint:
MODEL_SIZE = "large"  # "large" or "base"
CHECKPOINT_PATH = "/path/to/pretrain_rvm_large16_256.npz"  # <- point this at your downloaded RVM .npz

# Architecture presets matching the released JAX checkpoints.
RVM_CONFIGS = {
    "large": dict(
        VIT_EMBED_DIM=1024, VIT_NUM_LAYERS=24, VIT_NUM_HEADS=16, VIT_MLP_RATIO=4, VIT_PATCH_SIZE=16,
        RVM_NUM_LAYERS=4, RVM_NUM_HEADS=16, RVM_MLP_RATIO=4,
    ),
    "base": dict(
        VIT_EMBED_DIM=768,  VIT_NUM_LAYERS=12, VIT_NUM_HEADS=12, VIT_MLP_RATIO=4, VIT_PATCH_SIZE=16,
        RVM_NUM_LAYERS=4, RVM_NUM_HEADS=12, RVM_MLP_RATIO=4,
    ),
}
if MODEL_SIZE not in RVM_CONFIGS:
    raise ValueError(f"Unknown MODEL_SIZE={MODEL_SIZE!r}; expected one of {list(RVM_CONFIGS)}")

_cfg = RVM_CONFIGS[MODEL_SIZE]
VIT_EMBED_DIM  = _cfg["VIT_EMBED_DIM"]
VIT_NUM_LAYERS = _cfg["VIT_NUM_LAYERS"]
VIT_NUM_HEADS  = _cfg["VIT_NUM_HEADS"]
VIT_MLP_RATIO  = _cfg["VIT_MLP_RATIO"]
VIT_PATCH_SIZE = _cfg["VIT_PATCH_SIZE"]
RVM_NUM_LAYERS = _cfg["RVM_NUM_LAYERS"]
RVM_NUM_HEADS  = _cfg["RVM_NUM_HEADS"]
RVM_MLP_RATIO  = _cfg["RVM_MLP_RATIO"]


# =============================================================================
# Weight conversion utilities
# =============================================================================

def convert_dense_kernel(jax_kernel):
    """Convert JAX Dense kernel to PyTorch Linear weight.

    JAX Dense kernel: (input_dim, output_dim)
    PyTorch Linear weight: (output_dim, input_dim)
    """
    return torch.from_numpy(jax_kernel.T.copy())


def convert_attention_qkv_kernel(jax_kernel):
    """Convert JAX attention Q/K/V kernel to PyTorch format.

    JAX: (input_dim, num_heads, head_dim)
    PyTorch: (num_heads * head_dim, input_dim)
    """
    input_dim, num_heads, head_dim = jax_kernel.shape
    # Reshape to (input_dim, num_heads * head_dim) then transpose
    reshaped = jax_kernel.reshape(input_dim, num_heads * head_dim)
    return torch.from_numpy(reshaped.T.copy())


def convert_attention_qkv_bias(jax_bias):
    """Convert JAX attention Q/K/V bias to PyTorch format.

    JAX: (num_heads, head_dim)
    PyTorch: (num_heads * head_dim,)
    """
    return torch.from_numpy(jax_bias.reshape(-1).copy())


def convert_attention_out_kernel(jax_kernel):
    """Convert JAX attention output kernel to PyTorch format.

    JAX: (num_heads, head_dim, output_dim)
    PyTorch: (output_dim, num_heads * head_dim)
    """
    num_heads, head_dim, output_dim = jax_kernel.shape
    # Reshape to (num_heads * head_dim, output_dim) then transpose
    reshaped = jax_kernel.reshape(num_heads * head_dim, output_dim)
    return torch.from_numpy(reshaped.T.copy())


def convert_layernorm_scale(jax_scale):
    """Convert JAX LayerNorm scale to PyTorch weight."""
    return torch.from_numpy(jax_scale.copy())


def convert_layernorm_bias(jax_bias):
    """Convert JAX LayerNorm bias to PyTorch bias."""
    return torch.from_numpy(jax_bias.copy())


def convert_conv_kernel(jax_kernel, squeeze_temporal=False):
    """Convert JAX Conv kernel to PyTorch format.

    JAX: (T, H, W, in_channels, out_channels) for 3D conv, or (H, W, in_channels, out_channels) for 2D
    PyTorch: (out_channels, in_channels, T, H, W) or (out_channels, in_channels, H, W)

    If squeeze_temporal=True and T=1, convert to 2D by removing temporal dim.
    """
    if jax_kernel.ndim == 5:
        # 3D conv: (T, H, W, C_in, C_out)
        if squeeze_temporal and jax_kernel.shape[0] == 1:
            # Squeeze temporal: (1, H, W, C_in, C_out) -> (H, W, C_in, C_out) -> (C_out, C_in, H, W)
            jax_kernel = jax_kernel.squeeze(0)
            return torch.from_numpy(jax_kernel.transpose(3, 2, 0, 1).copy())
        else:
            # Keep 3D: (T, H, W, C_in, C_out) -> (C_out, C_in, T, H, W)
            return torch.from_numpy(jax_kernel.transpose(4, 3, 0, 1, 2).copy())
    elif jax_kernel.ndim == 4:
        # 2D conv: (H, W, C_in, C_out) -> (C_out, C_in, H, W)
        return torch.from_numpy(jax_kernel.transpose(3, 2, 0, 1).copy())
    else:
        raise ValueError(f"Unexpected conv kernel ndim: {jax_kernel.ndim}")


# =============================================================================
# Build PyTorch model matching checkpoint structure
# =============================================================================

class ImprovedMultiHeadDotProductAttention(nn.Module):
    """Multi-head attention with separate Q, K, V projections (matching JAX structure).

    This allows direct weight loading from JAX checkpoint without reshaping.
    """
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

        # Project and reshape
        q = self.query(q).view(B, N, self.num_heads, self.head_dim)
        k = self.key(k).view(B, S, self.num_heads, self.head_dim)
        v = self.value(v).view(B, S, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bnhd,bshd->bhns', q * scale, k)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhns,bshd->bnhd', attn, v)
        out = out.reshape(B, N, self.embed_dim)
        out = self.out(out)

        if need_weights:
            return out, attn
        return out, None


class TransformerMLP(nn.Module):
    """MLP block matching JAX structure."""
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


class PreNormBlock(nn.Module):
    """Transformer block with pre-norm (matching JAX encoder structure)."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = ImprovedMultiHeadDotProductAttention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = TransformerMLP(embed_dim, mlp_ratio)

    def forward(self, x):
        # Self-attention with pre-norm
        x_norm = self.attention_norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.mlp_norm(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block matching JAX rnn_core/transformer structure.

    Order: cross-attention -> MLP -> self-attention
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        # Cross-attention
        self.ca_attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ca_attention = ImprovedMultiHeadDotProductAttention(embed_dim, num_heads)

        # MLP
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = TransformerMLP(embed_dim, mlp_ratio)

        # Self-attention
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = ImprovedMultiHeadDotProductAttention(embed_dim, num_heads)

    def forward(self, x, x_kv):
        # Cross-attention
        x_norm = self.ca_attention_norm(x)
        ca_out, _ = self.ca_attention(x_norm, x_kv, x_kv)
        x = x + ca_out

        # MLP
        x = x + self.mlp(self.mlp_norm(x))

        # Self-attention
        x_norm = self.attention_norm(x)
        sa_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + sa_out

        return x


class CrossAttentionTransformer(nn.Module):
    """Stack of cross-attention blocks (matching JAX rnn_core/transformer)."""
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4):
        super().__init__()
        self.xa_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, x_kv):
        for block in self.xa_blocks:
            x = block(x, x_kv)
        return self.output_norm(x)


class GatedTransformerCore(nn.Module):
    """RVM's gated transformer core (matching JAX rnn_core structure)."""
    def __init__(self, embed_dim, num_layers=4, num_heads=16, mlp_ratio=4):
        super().__init__()
        self.embed_dim = embed_dim

        # Gating (no bias, as in original)
        self.input_update = nn.Linear(embed_dim, embed_dim, bias=False)
        self.input_reset = nn.Linear(embed_dim, embed_dim, bias=False)
        self.state_update = nn.Linear(embed_dim, embed_dim, bias=False)
        self.state_reset = nn.Linear(embed_dim, embed_dim, bias=False)

        # State normalization (scale only, no bias in original)
        self.state_layer_norm = nn.LayerNorm(embed_dim, eps=1e-6, elementwise_affine=True)
        # Note: JAX checkpoint only has 'scale', no 'bias' for state_layer_norm
        # We'll set bias to zero after loading

        # Cross-attention transformer
        self.transformer = CrossAttentionTransformer(embed_dim, num_layers, num_heads, mlp_ratio)

    def forward(self, inputs, state):
        """Single step of the gated transformer core."""
        update_gate = torch.sigmoid(self.input_update(inputs) + self.state_update(state))
        reset_gate = torch.sigmoid(self.input_reset(inputs) + self.state_reset(state))

        gated_state = reset_gate * self.state_layer_norm(state)
        h = self.transformer(inputs, gated_state)

        output = (1 - update_gate) * state + update_gate * h
        return output, output


class ViTEncoder(nn.Module):
    """ViT encoder (matching JAX encoder structure)."""
    def __init__(self, embed_dim, num_layers, num_heads, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.LayerNorm_0 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm_0(x)


class PatchEmbedding(nn.Module):
    """Patch embedding (matching JAX tokenizer/patch_embedding).

    Note: JAX checkpoint uses (1, 16, 16) patch which is technically 3D conv
    with temporal=1. We use 2D conv and squeeze the temporal dim when loading.
    """
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()
        # Use 2D conv (matching typical ViT)
        self.Conv_0 = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        return self.Conv_0(x)


class RVMModel(nn.Module):
    """Complete RVM model: Tokenizer + Encoder + RNN Core.

    This matches the structure in the JAX checkpoint.
    """
    def __init__(
        self,
        embed_dim=1024,
        patch_size=16,
        encoder_layers=24,
        encoder_heads=16,
        encoder_mlp_ratio=4,
        rnn_layers=4,
        rnn_heads=16,
        rnn_mlp_ratio=4,
    ):
        super().__init__()

        # Tokenizer (2D patch embedding, applied per frame)
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Encoder
        self.encoder = ViTEncoder(embed_dim, encoder_layers, encoder_heads, encoder_mlp_ratio)

        # RNN Core
        self.rnn_core = GatedTransformerCore(embed_dim, rnn_layers, rnn_heads, rnn_mlp_ratio)

    def forward(self, frames, state=None):
        """
        Args:
            frames: (B, T, C, H, W) video frames
            state: Optional previous state

        Returns:
            features: (B, T, N+1, D) encoded features
            state: Updated state
        """
        B, T, C, H, W = frames.shape

        # Process each frame
        outputs = []
        for t in range(T):
            # Tokenize: (B, C, H, W) -> (B, D, h, w)
            tokens = self.patch_embedding(frames[:, t])
            # Reshape: (B, D, h, w) -> (B, h*w, D)
            tokens = tokens.flatten(2).transpose(1, 2)

            # Add CLS token
            cls = self.cls_token.expand(B, 1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            # Encode
            encoded = self.encoder(tokens)

            # RNN step
            if state is None:
                state = torch.zeros_like(encoded)
            output, state = self.rnn_core(encoded, state)
            outputs.append(output)

        return torch.stack(outputs, dim=1), state


# =============================================================================
# Weight loading functions
# =============================================================================

def load_attention_weights(attn_module, jax_dict, prefix):
    """Load weights for ImprovedMultiHeadDotProductAttention."""
    # Query
    attn_module.query.weight.data = convert_attention_qkv_kernel(jax_dict[f'{prefix}/query/kernel'])
    attn_module.query.bias.data = convert_attention_qkv_bias(jax_dict[f'{prefix}/query/bias'])

    # Key
    attn_module.key.weight.data = convert_attention_qkv_kernel(jax_dict[f'{prefix}/key/kernel'])
    attn_module.key.bias.data = convert_attention_qkv_bias(jax_dict[f'{prefix}/key/bias'])

    # Value
    attn_module.value.weight.data = convert_attention_qkv_kernel(jax_dict[f'{prefix}/value/kernel'])
    attn_module.value.bias.data = convert_attention_qkv_bias(jax_dict[f'{prefix}/value/bias'])

    # Output
    attn_module.out.weight.data = convert_attention_out_kernel(jax_dict[f'{prefix}/out/kernel'])
    attn_module.out.bias.data = torch.from_numpy(jax_dict[f'{prefix}/out/bias'].copy())


def load_mlp_weights(mlp_module, jax_dict, prefix):
    """Load weights for TransformerMLP."""
    mlp_module.dense_in.weight.data = convert_dense_kernel(jax_dict[f'{prefix}/dense_in/kernel'])
    mlp_module.dense_in.bias.data = torch.from_numpy(jax_dict[f'{prefix}/dense_in/bias'].copy())
    mlp_module.dense_out.weight.data = convert_dense_kernel(jax_dict[f'{prefix}/dense_out/kernel'])
    mlp_module.dense_out.bias.data = torch.from_numpy(jax_dict[f'{prefix}/dense_out/bias'].copy())


def load_layernorm_weights(ln_module, jax_dict, prefix, has_bias=True):
    """Load weights for LayerNorm."""
    ln_module.weight.data = convert_layernorm_scale(jax_dict[f'{prefix}/scale'])
    if has_bias and f'{prefix}/bias' in jax_dict:
        ln_module.bias.data = convert_layernorm_bias(jax_dict[f'{prefix}/bias'])
    else:
        ln_module.bias.data.zero_()


def load_encoder_block_weights(block, jax_dict, prefix):
    """Load weights for a PreNormBlock."""
    # Attention norm
    load_layernorm_weights(block.attention_norm, jax_dict, f'{prefix}/attention_norm')
    # Attention
    load_attention_weights(block.attention, jax_dict, f'{prefix}/attention')
    # MLP norm
    load_layernorm_weights(block.mlp_norm, jax_dict, f'{prefix}/mlp_norm')
    # MLP
    load_mlp_weights(block.mlp, jax_dict, f'{prefix}/mlp')


def load_cross_attention_block_weights(block, jax_dict, prefix):
    """Load weights for a CrossAttentionBlock."""
    # Cross-attention norm
    load_layernorm_weights(block.ca_attention_norm, jax_dict, f'{prefix}/ca_attention_norm')
    # Cross-attention
    load_attention_weights(block.ca_attention, jax_dict, f'{prefix}/ca_attention')
    # MLP norm
    load_layernorm_weights(block.mlp_norm, jax_dict, f'{prefix}/mlp_norm')
    # MLP
    load_mlp_weights(block.mlp, jax_dict, f'{prefix}/mlp')
    # Self-attention norm
    load_layernorm_weights(block.attention_norm, jax_dict, f'{prefix}/attention_norm')
    # Self-attention
    load_attention_weights(block.attention, jax_dict, f'{prefix}/attention')


def load_rvm_checkpoint(model, checkpoint_path):
    """Load RVM checkpoint into model."""
    print(f"Loading checkpoint from {checkpoint_path}")
    jax_dict = np.load(checkpoint_path, allow_pickle=True)

    loaded_keys = set()

    # 1. Load CLS token
    model.cls_token.data = torch.from_numpy(jax_dict['cls_token'].copy())
    loaded_keys.add('cls_token')
    print(f"  Loaded cls_token: {model.cls_token.shape}")

    # 2. Load patch embedding (squeeze temporal dim since checkpoint has (1,16,16) but we use 2D conv)
    conv_kernel = jax_dict['tokenizer/patch_embedding/Conv_0/kernel']
    model.patch_embedding.Conv_0.weight.data = convert_conv_kernel(conv_kernel, squeeze_temporal=True)
    model.patch_embedding.Conv_0.bias.data = torch.from_numpy(
        jax_dict['tokenizer/patch_embedding/Conv_0/bias'].copy()
    )
    loaded_keys.add('tokenizer/patch_embedding/Conv_0/kernel')
    loaded_keys.add('tokenizer/patch_embedding/Conv_0/bias')
    print(f"  Loaded patch_embedding: {model.patch_embedding.Conv_0.weight.shape}")

    # 3. Load encoder layers
    for i in range(len(model.encoder.layers)):
        prefix = f'encoder/layers_{i}'
        load_encoder_block_weights(model.encoder.layers[i], jax_dict, prefix)
        loaded_keys.add(f'{prefix}/attention_norm/scale')
        loaded_keys.add(f'{prefix}/attention_norm/bias')
        loaded_keys.add(f'{prefix}/attention/query/kernel')
        loaded_keys.add(f'{prefix}/attention/query/bias')
        loaded_keys.add(f'{prefix}/attention/key/kernel')
        loaded_keys.add(f'{prefix}/attention/key/bias')
        loaded_keys.add(f'{prefix}/attention/value/kernel')
        loaded_keys.add(f'{prefix}/attention/value/bias')
        loaded_keys.add(f'{prefix}/attention/out/kernel')
        loaded_keys.add(f'{prefix}/attention/out/bias')
        loaded_keys.add(f'{prefix}/mlp_norm/scale')
        loaded_keys.add(f'{prefix}/mlp_norm/bias')
        loaded_keys.add(f'{prefix}/mlp/dense_in/kernel')
        loaded_keys.add(f'{prefix}/mlp/dense_in/bias')
        loaded_keys.add(f'{prefix}/mlp/dense_out/kernel')
        loaded_keys.add(f'{prefix}/mlp/dense_out/bias')
    print(f"  Loaded {len(model.encoder.layers)} encoder layers")

    # 4. Load encoder final LayerNorm
    load_layernorm_weights(model.encoder.LayerNorm_0, jax_dict, 'encoder/LayerNorm_0')
    loaded_keys.add('encoder/LayerNorm_0/scale')
    loaded_keys.add('encoder/LayerNorm_0/bias')
    print(f"  Loaded encoder LayerNorm")

    # 5. Load RNN Core gating weights
    model.rnn_core.input_update.weight.data = convert_dense_kernel(jax_dict['rnn_core/input_update/kernel'])
    model.rnn_core.input_reset.weight.data = convert_dense_kernel(jax_dict['rnn_core/input_reset/kernel'])
    model.rnn_core.state_update.weight.data = convert_dense_kernel(jax_dict['rnn_core/state_update/kernel'])
    model.rnn_core.state_reset.weight.data = convert_dense_kernel(jax_dict['rnn_core/state_reset/kernel'])
    loaded_keys.add('rnn_core/input_update/kernel')
    loaded_keys.add('rnn_core/input_reset/kernel')
    loaded_keys.add('rnn_core/state_update/kernel')
    loaded_keys.add('rnn_core/state_reset/kernel')
    print(f"  Loaded rnn_core gating weights")

    # 6. Load state layer norm (only scale, no bias in original)
    model.rnn_core.state_layer_norm.weight.data = convert_layernorm_scale(
        jax_dict['rnn_core/state_layer_norm/scale']
    )
    model.rnn_core.state_layer_norm.bias.data.zero_()  # No bias in original
    loaded_keys.add('rnn_core/state_layer_norm/scale')
    print(f"  Loaded rnn_core state_layer_norm")

    # 7. Load RNN Core transformer blocks
    for i in range(len(model.rnn_core.transformer.xa_blocks)):
        prefix = f'rnn_core/transformer/xa_blocks_{i}'
        load_cross_attention_block_weights(model.rnn_core.transformer.xa_blocks[i], jax_dict, prefix)
        # Track loaded keys
        for subprefix in ['ca_attention_norm', 'ca_attention', 'mlp_norm', 'mlp', 'attention_norm', 'attention']:
            if 'norm' in subprefix:
                loaded_keys.add(f'{prefix}/{subprefix}/scale')
                loaded_keys.add(f'{prefix}/{subprefix}/bias')
            elif 'mlp' in subprefix:
                loaded_keys.add(f'{prefix}/{subprefix}/dense_in/kernel')
                loaded_keys.add(f'{prefix}/{subprefix}/dense_in/bias')
                loaded_keys.add(f'{prefix}/{subprefix}/dense_out/kernel')
                loaded_keys.add(f'{prefix}/{subprefix}/dense_out/bias')
            else:
                for qkv in ['query', 'key', 'value', 'out']:
                    loaded_keys.add(f'{prefix}/{subprefix}/{qkv}/kernel')
                    loaded_keys.add(f'{prefix}/{subprefix}/{qkv}/bias')
    print(f"  Loaded {len(model.rnn_core.transformer.xa_blocks)} rnn_core transformer blocks")

    # 8. Load RNN Core output norm
    load_layernorm_weights(model.rnn_core.transformer.output_norm, jax_dict, 'rnn_core/transformer/output_norm')
    loaded_keys.add('rnn_core/transformer/output_norm/scale')
    loaded_keys.add('rnn_core/transformer/output_norm/bias')
    print(f"  Loaded rnn_core transformer output_norm")

    # Check for unused keys (decoder, mask_token, etc. - expected to be unused)
    all_keys = set(jax_dict.keys())
    unused_keys = all_keys - loaded_keys

    # Filter expected unused keys (decoder, mask_token, delta_embedder, detokenizer, decoder_embedder)
    expected_unused_prefixes = ['decoder/', 'mask_token', 'delta_embedder/', 'detokenizer/', 'decoder_embedder/']
    unexpected_unused = [k for k in unused_keys if not any(k.startswith(p) for p in expected_unused_prefixes)]

    print(f"\n  Total keys in checkpoint: {len(all_keys)}")
    print(f"  Keys loaded: {len(loaded_keys)}")
    print(f"  Keys unused (expected - decoder etc.): {len(unused_keys) - len(unexpected_unused)}")

    if unexpected_unused:
        print(f"\n  WARNING: Unexpected unused keys:")
        for k in sorted(unexpected_unused):
            print(f"    {k}")
    else:
        print(f"\n  SUCCESS: All encoder and rnn_core weights loaded!")

    return model


# =============================================================================
# Convert to RVMCoreThenReadout-compatible format
# =============================================================================

def fuse_qkv_weights(q_weight, k_weight, v_weight):
    """Fuse separate Q, K, V weights into nn.MultiheadAttention's in_proj_weight.

    nn.MultiheadAttention expects in_proj_weight of shape (3 * embed_dim, embed_dim)
    with Q, K, V stacked in that order.
    """
    # All weights should be (embed_dim, embed_dim)
    return torch.cat([q_weight, k_weight, v_weight], dim=0)


def fuse_qkv_bias(q_bias, k_bias, v_bias):
    """Fuse separate Q, K, V biases into nn.MultiheadAttention's in_proj_bias."""
    return torch.cat([q_bias, k_bias, v_bias], dim=0)


def convert_to_wrapper_format(state_dict, embed_dim):
    """Convert RVMModel state_dict to RVMClassificationWrapper-compatible format.

    This handles the key remapping and Q/K/V fusion needed to load weights into
    RVMCoreThenReadout (which uses nn.MultiheadAttention with fused projections).

    Key mappings for rnn_core -> rvm_core:
    - rnn_core.transformer.xa_blocks.{i} -> rvm_core.transformer.layers.{i}
    - ca_attention -> ca_attn (cross-attention)
    - attention -> sa_attn (self-attention)
    - query/key/value.weight + out.weight -> in_proj_weight + out_proj.weight
    - mlp.dense_in -> mlp.fc1
    - mlp.dense_out -> mlp.fc2
    - ca_attention_norm -> ca_norm
    - attention_norm -> sa_norm
    """
    wrapper_dict = OrderedDict()

    # Process encoder weights (mostly direct mapping with minor key changes)
    for key, value in state_dict.items():
        if key.startswith('encoder.') or key in ['cls_token', 'patch_embedding.Conv_0.weight', 'patch_embedding.Conv_0.bias']:
            # Keep encoder keys as-is (RVMViTEncoder uses same structure)
            wrapper_dict[key] = value

    # Process rnn_core weights -> rvm_core (with Q/K/V fusion)
    # First, collect all the rnn_core weights organized by layer and component
    rnn_core_weights = {}
    for key, value in state_dict.items():
        if key.startswith('rnn_core.'):
            rnn_core_weights[key] = value

    # Map gating weights (direct mapping, just rename rnn_core -> rvm_core)
    gating_keys = ['input_update', 'input_reset', 'state_update', 'state_reset']
    for gk in gating_keys:
        old_key = f'rnn_core.{gk}.weight'
        new_key = f'rvm_core.{gk}.weight'
        if old_key in rnn_core_weights:
            wrapper_dict[new_key] = rnn_core_weights[old_key]

    # Map state_layer_norm
    wrapper_dict['rvm_core.state_layer_norm.weight'] = rnn_core_weights['rnn_core.state_layer_norm.weight']
    wrapper_dict['rvm_core.state_layer_norm.bias'] = rnn_core_weights['rnn_core.state_layer_norm.bias']

    # Map transformer blocks (xa_blocks -> layers)
    # Find number of layers
    layer_indices = set()
    for key in rnn_core_weights:
        if 'xa_blocks.' in key:
            # Extract layer index from keys like 'rnn_core.transformer.xa_blocks.0.ca_attention.query.weight'
            parts = key.split('.')
            idx = int(parts[3])
            layer_indices.add(idx)

    for layer_idx in sorted(layer_indices):
        old_prefix = f'rnn_core.transformer.xa_blocks.{layer_idx}'
        new_prefix = f'rvm_core.transformer.layers.{layer_idx}'

        # Cross-attention (ca_attention -> ca_attn)
        # Fuse Q/K/V into in_proj_weight
        q_w = rnn_core_weights[f'{old_prefix}.ca_attention.query.weight']
        k_w = rnn_core_weights[f'{old_prefix}.ca_attention.key.weight']
        v_w = rnn_core_weights[f'{old_prefix}.ca_attention.value.weight']
        wrapper_dict[f'{new_prefix}.ca_attn.in_proj_weight'] = fuse_qkv_weights(q_w, k_w, v_w)

        q_b = rnn_core_weights[f'{old_prefix}.ca_attention.query.bias']
        k_b = rnn_core_weights[f'{old_prefix}.ca_attention.key.bias']
        v_b = rnn_core_weights[f'{old_prefix}.ca_attention.value.bias']
        wrapper_dict[f'{new_prefix}.ca_attn.in_proj_bias'] = fuse_qkv_bias(q_b, k_b, v_b)

        # Output projection
        wrapper_dict[f'{new_prefix}.ca_attn.out_proj.weight'] = rnn_core_weights[f'{old_prefix}.ca_attention.out.weight']
        wrapper_dict[f'{new_prefix}.ca_attn.out_proj.bias'] = rnn_core_weights[f'{old_prefix}.ca_attention.out.bias']

        # Cross-attention norm (ca_attention_norm -> ca_norm)
        wrapper_dict[f'{new_prefix}.ca_norm.weight'] = rnn_core_weights[f'{old_prefix}.ca_attention_norm.weight']
        wrapper_dict[f'{new_prefix}.ca_norm.bias'] = rnn_core_weights[f'{old_prefix}.ca_attention_norm.bias']

        # Self-attention (attention -> sa_attn)
        # Fuse Q/K/V into in_proj_weight
        q_w = rnn_core_weights[f'{old_prefix}.attention.query.weight']
        k_w = rnn_core_weights[f'{old_prefix}.attention.key.weight']
        v_w = rnn_core_weights[f'{old_prefix}.attention.value.weight']
        wrapper_dict[f'{new_prefix}.sa_attn.in_proj_weight'] = fuse_qkv_weights(q_w, k_w, v_w)

        q_b = rnn_core_weights[f'{old_prefix}.attention.query.bias']
        k_b = rnn_core_weights[f'{old_prefix}.attention.key.bias']
        v_b = rnn_core_weights[f'{old_prefix}.attention.value.bias']
        wrapper_dict[f'{new_prefix}.sa_attn.in_proj_bias'] = fuse_qkv_bias(q_b, k_b, v_b)

        # Output projection
        wrapper_dict[f'{new_prefix}.sa_attn.out_proj.weight'] = rnn_core_weights[f'{old_prefix}.attention.out.weight']
        wrapper_dict[f'{new_prefix}.sa_attn.out_proj.bias'] = rnn_core_weights[f'{old_prefix}.attention.out.bias']

        # Self-attention norm (attention_norm -> sa_norm)
        wrapper_dict[f'{new_prefix}.sa_norm.weight'] = rnn_core_weights[f'{old_prefix}.attention_norm.weight']
        wrapper_dict[f'{new_prefix}.sa_norm.bias'] = rnn_core_weights[f'{old_prefix}.attention_norm.bias']

        # MLP (dense_in/dense_out -> fc1/fc2)
        wrapper_dict[f'{new_prefix}.mlp.fc1.weight'] = rnn_core_weights[f'{old_prefix}.mlp.dense_in.weight']
        wrapper_dict[f'{new_prefix}.mlp.fc1.bias'] = rnn_core_weights[f'{old_prefix}.mlp.dense_in.bias']
        wrapper_dict[f'{new_prefix}.mlp.fc2.weight'] = rnn_core_weights[f'{old_prefix}.mlp.dense_out.weight']
        wrapper_dict[f'{new_prefix}.mlp.fc2.bias'] = rnn_core_weights[f'{old_prefix}.mlp.dense_out.bias']

        # MLP norm
        wrapper_dict[f'{new_prefix}.mlp_norm.weight'] = rnn_core_weights[f'{old_prefix}.mlp_norm.weight']
        wrapper_dict[f'{new_prefix}.mlp_norm.bias'] = rnn_core_weights[f'{old_prefix}.mlp_norm.bias']

    # Map transformer output_norm
    wrapper_dict['rvm_core.transformer.output_norm.weight'] = rnn_core_weights['rnn_core.transformer.output_norm.weight']
    wrapper_dict['rvm_core.transformer.output_norm.bias'] = rnn_core_weights['rnn_core.transformer.output_norm.bias']

    return wrapper_dict


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Create model
    print(f"Creating RVM-{MODEL_SIZE.capitalize()} model...")
    model = RVMModel(
        embed_dim=VIT_EMBED_DIM,
        patch_size=VIT_PATCH_SIZE,
        encoder_layers=VIT_NUM_LAYERS,
        encoder_heads=VIT_NUM_HEADS,
        encoder_mlp_ratio=VIT_MLP_RATIO,
        rnn_layers=RVM_NUM_LAYERS,
        rnn_heads=RVM_NUM_HEADS,
        rnn_mlp_ratio=RVM_MLP_RATIO,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # Load checkpoint
    model = load_rvm_checkpoint(model, CHECKPOINT_PATH)

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        # Dummy input: batch=2, frames=4, channels=3, height=256, width=256
        x = torch.randn(2, 4, 3, 256, 256)
        features, state = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output features shape: {features.shape}")
        print(f"  Output state shape: {state.shape}")

    # Save the RVMCoreThenReadout-compatible PyTorch checkpoint
    # (fuses Q/K/V weights for nn.MultiheadAttention; this is the variant the
    # inference scripts and training pipeline expect when an RVM checkpoint is needed).
    state_dict = model.state_dict()
    wrapper_state_dict = convert_to_wrapper_format(state_dict, VIT_EMBED_DIM)
    pytorch_wrapper_path = CHECKPOINT_PATH.replace('.npz', '_pytorch_wrapper.pth')
    print(f"\nSaving RVMCoreThenReadout-compatible checkpoint to {pytorch_wrapper_path}")
    torch.save(wrapper_state_dict, pytorch_wrapper_path)

    wrapper_enc_keys = [k for k in wrapper_state_dict.keys() if k.startswith('encoder.')]
    wrapper_core_keys = [k for k in wrapper_state_dict.keys() if k.startswith('rvm_core.')]
    print(f"  {len(wrapper_enc_keys)} encoder + {len(wrapper_core_keys)} rvm_core = {len(wrapper_state_dict)} parameters")

    print("\nDone!")
