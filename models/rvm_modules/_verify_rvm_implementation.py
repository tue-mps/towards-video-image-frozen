"""Verify our RVM implementation against the reference VideoSiamMAE from the RVM colab.

This script:
1. Builds the reference VideoSiamMAE model (from the RVM colab notebook)
2. Builds our RVMViTEncoder + GatedTransformerCore
3. Loads the same JAX checkpoint into both
4. Compares outputs on the same random input

Reference: https://github.com/google-deepmind/representations4d/blob/main/colabs/rvm_inference_demo.ipynb

Usage:
    python models/rvm_modules/verify_rvm_implementation.py
"""

CHECKPOINT_PATH = "/path/to/pretrain_rvm_large16_256.npz"  # <- the original RVM .npz; its *_pytorch_wrapper.pth must already exist next to it

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Reference implementation: VideoSiamMAE from the RVM colab
# =============================================================================


def get_mae_sinusoid_encoding_table(n_position, d_hid, dtype=torch.float32):
    """MAE-style sinusoidal positional encoding table."""
    def get_position_angle_vec(position):
        return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                               for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.tensor(sinusoid_table, dtype=dtype)[None, ...]


class RefPatchEmbedding(nn.Module):
    """Reference patch embedding (channels-last convention like JAX)."""
    def __init__(self, patch_size=(16, 16), num_features=1024):
        super().__init__()
        self.patch_size = patch_size
        self.Conv_0 = nn.Conv2d(3, num_features, kernel_size=patch_size,
                                stride=patch_size, padding=0)

    def forward(self, x):
        # Reference uses channels-last (B, H, W, C), we use channels-first
        return self.Conv_0(x).permute(0, 2, 3, 1)  # (B, h, w, D)


class RefSincosPosEmb(nn.Module):
    """Sinusoidal positional embedding module (from colab)."""
    def __init__(self, base_token_shape=None):
        super().__init__()
        self.base_token_shape = base_token_shape

    def forward(self, tokens_shape, device):
        d = tokens_shape[-1]
        h, w = (self.base_token_shape if self.base_token_shape
                else (tokens_shape[-3], tokens_shape[-2]))
        posenc = get_mae_sinusoid_encoding_table(h * w, d)
        posenc = posenc.view(1, h, w, d)
        *b, tokens_h, tokens_w, _ = tokens_shape
        for _ in range(len(b) - 1):
            posenc = posenc.expand(*b, -1, -1, -1)

        if tokens_h != h or tokens_w != w:
            posenc = posenc.view(-1, h, w, d)
            posenc = F.interpolate(posenc.permute(0, 3, 1, 2),
                                   size=(tokens_h, tokens_w),
                                   mode='bicubic', align_corners=False
                                   ).permute(0, 2, 3, 1)
            posenc = posenc.view(*b, tokens_h, tokens_w, d)
        return posenc.to(device)


class RefTokenizer(nn.Module):
    def __init__(self, patch_embedding, posenc):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.posenc = posenc

    def forward(self, x):
        tokens = self.patch_embedding(x)  # (B, h, w, D)
        posenc = self.posenc(tokens.shape, tokens.device)
        return tokens + posenc


class RefImprovedMHDPA(nn.Module):
    """Multi-head attention (from colab)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs_q, inputs_k=None, inputs_v=None):
        if inputs_k is None:
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        B, N, _ = inputs_q.shape
        S = inputs_k.shape[1]

        q = self.query(inputs_q).view(B, N, self.num_heads, self.head_dim)
        k = self.key(inputs_k).view(B, S, self.num_heads, self.head_dim)
        v = self.value(inputs_v).view(B, S, self.num_heads, self.head_dim)

        scale = self.head_dim ** -0.5
        attn = torch.einsum('bqhd,bkhd->bhqk', q * scale, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(B, N, self.embed_dim)
        return self.out(out)


class RefTransformerMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=None):
        super().__init__()
        self.dense_in = nn.Linear(input_dim, hidden_size or (4 * input_dim))
        self.dense_out = nn.Linear(hidden_size or (4 * input_dim), input_dim)

    def forward(self, x):
        return self.dense_out(F.gelu(self.dense_in(x)))


class RefPreNormBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = RefImprovedMHDPA(embed_dim, num_heads)
        self.mlp = RefTransformerMLP(embed_dim, embed_dim * mlp_ratio)

    def forward(self, x):
        norm_x = self.attention_norm(x)
        x = x + self.attention(norm_x)
        norm_x = self.mlp_norm(x)
        return x + self.mlp(norm_x)


class RefTransformer(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, mlp_size):
        super().__init__()
        self.layers = nn.ModuleList([
            RefPreNormBlock(hidden_size, num_heads, mlp_size // hidden_size)
            for _ in range(num_layers)
        ])
        self.LayerNorm_0 = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.LayerNorm_0(x)


class RefCrossAttentionBlock(nn.Module):
    """Cross-attention block: CA -> MLP -> SA (from colab)."""
    def __init__(self, num_heads, num_feats, mlp_dim):
        super().__init__()
        self.ca_attention_norm = nn.LayerNorm(num_feats, eps=1e-6)
        self.ca_attention = RefImprovedMHDPA(num_feats, num_heads)
        self.mlp_norm = nn.LayerNorm(num_feats, eps=1e-6)
        self.mlp = RefTransformerMLP(num_feats, mlp_dim)
        self.attention_norm = nn.LayerNorm(num_feats, eps=1e-6)
        self.attention = RefImprovedMHDPA(num_feats, num_heads)

    def forward(self, x, x_kv):
        x = x + self.ca_attention(inputs_q=self.ca_attention_norm(x),
                                  inputs_k=x_kv, inputs_v=x_kv)
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.attention(self.attention_norm(x))
        return x


class RefCrossAttentionTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, num_feats, mlp_dim):
        super().__init__()
        self.xa_blocks = nn.ModuleList([
            RefCrossAttentionBlock(num_heads, num_feats, mlp_dim)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(num_feats, eps=1e-6)

    def forward(self, inputs, inputs_kv):
        x = inputs
        for block in self.xa_blocks:
            x = block(x, inputs_kv)
        return self.output_norm(x)


class RefGatedTransformerCore(nn.Module):
    def __init__(self, transformer, token_dim):
        super().__init__()
        self.transformer = transformer
        self.token_dim = token_dim
        self.state_layer_norm = nn.LayerNorm(token_dim, eps=1e-6)
        self.input_update = nn.Linear(token_dim, token_dim, bias=False)
        self.input_reset = nn.Linear(token_dim, token_dim, bias=False)
        self.state_update = nn.Linear(token_dim, token_dim, bias=False)
        self.state_reset = nn.Linear(token_dim, token_dim, bias=False)

    def initializer(self, inputs, batch_shape):
        shape = inputs.shape[-2:]
        return torch.zeros(batch_shape + shape, dtype=inputs.dtype,
                           device=inputs.device)

    def forward(self, inputs, state):
        update_gate = torch.sigmoid(self.input_update(inputs) +
                                    self.state_update(state))
        reset_gate = torch.sigmoid(self.input_reset(inputs) +
                                   self.state_reset(state))
        h = self.transformer(inputs, inputs_kv=reset_gate *
                             self.state_layer_norm(state))
        output = (1 - update_gate) * state + update_gate * h
        return output, output


class RefVideoSiamMAE(nn.Module):
    """Reference VideoSiamMAE from the RVM colab."""
    def __init__(self, tokenizer, encoder, rnn_core, latent_emb_dim=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.rnn_core = rnn_core
        self.latent_emb_dim = latent_emb_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_emb_dim) * 0.02)

    def forward(self, frame, state=None):
        """Process a single frame.

        Args:
            frame: (B, C, H, W) single frame (channels-first)
            state: optional previous state

        Returns:
            features: (B, N+1, D)
            state: updated state
        """
        # Tokenize: (B, C, H, W) -> (B, h, w, D) -> (B, h*w, D)
        import einops
        frame_tokens = self.tokenizer(frame)
        frame_tokens = einops.rearrange(frame_tokens, 'b h w d -> b (h w) d')

        # Prepend CLS
        B = frame_tokens.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        frame_tokens = torch.cat([cls_token, frame_tokens], dim=1)

        # Encode
        encoded_frame_tokens = self.encoder(frame_tokens)

        # RNN step
        if state is None:
            state = self.rnn_core.initializer(encoded_frame_tokens, batch_shape=(B,))
        features, state = self.rnn_core(encoded_frame_tokens, state)
        return features, state


# =============================================================================
# Build reference model from config
# =============================================================================

def build_reference_model(embed_dim=1024, patch_size=16, num_encoder_layers=24,
                          num_heads=16, mlp_ratio=4, rnn_layers=4,
                          base_token_shape=(16, 16)):
    """Build the reference VideoSiamMAE model."""
    patch_emb = RefPatchEmbedding(patch_size=(patch_size, patch_size),
                                  num_features=embed_dim)
    posenc = RefSincosPosEmb(base_token_shape=base_token_shape)
    tokenizer = RefTokenizer(patch_emb, posenc)

    encoder = RefTransformer(
        num_layers=num_encoder_layers,
        hidden_size=embed_dim,
        num_heads=num_heads,
        mlp_size=embed_dim * mlp_ratio,
    )

    rnn_transformer = RefCrossAttentionTransformer(
        num_layers=rnn_layers,
        num_heads=num_heads,
        num_feats=embed_dim,
        mlp_dim=embed_dim * mlp_ratio,
    )
    rnn_core = RefGatedTransformerCore(rnn_transformer, token_dim=embed_dim)

    model = RefVideoSiamMAE(tokenizer, encoder, rnn_core,
                             latent_emb_dim=embed_dim)
    return model


# =============================================================================
# Load JAX checkpoint into reference model using colab's flax_to_torch approach
# =============================================================================

def load_jax_into_reference(model, checkpoint_path):
    """Load JAX .npz checkpoint into the reference VideoSiamMAE model.

    Uses the same weight conversion logic as load_rvm_ckpt.py.
    """
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from scripts_support.load_rvm_ckpt import (
        load_rvm_checkpoint, RVMModel
    )

    # Build the intermediate RVMModel (from load_rvm_ckpt.py) which has
    # the same structure and proven checkpoint loading
    intermediate = RVMModel(
        embed_dim=1024, patch_size=16,
        encoder_layers=24, encoder_heads=16, encoder_mlp_ratio=4,
        rnn_layers=4, rnn_heads=16, rnn_mlp_ratio=4,
    )
    intermediate = load_rvm_checkpoint(intermediate, checkpoint_path)
    src = intermediate.state_dict()

    # Now copy weights from intermediate -> reference model
    # The structures are identical, just different nesting
    dst = model.state_dict()

    # CLS token
    dst['cls_token'] = src['cls_token'].unsqueeze(0)  # (1, D) -> (1, 1, D)

    # Patch embedding
    dst['tokenizer.patch_embedding.Conv_0.weight'] = src['patch_embedding.Conv_0.weight']
    dst['tokenizer.patch_embedding.Conv_0.bias'] = src['patch_embedding.Conv_0.bias']

    # Encoder layers
    for i in range(24):
        sp = f'encoder.layers.{i}'
        dp = f'encoder.layers.{i}'
        for suffix in [
            'attention_norm.weight', 'attention_norm.bias',
            'attention.query.weight', 'attention.query.bias',
            'attention.key.weight', 'attention.key.bias',
            'attention.value.weight', 'attention.value.bias',
            'attention.out.weight', 'attention.out.bias',
            'mlp_norm.weight', 'mlp_norm.bias',
            'mlp.dense_in.weight', 'mlp.dense_in.bias',
            'mlp.dense_out.weight', 'mlp.dense_out.bias',
        ]:
            dst[f'{dp}.{suffix}'] = src[f'{sp}.{suffix}']

    # Encoder final norm
    dst['encoder.LayerNorm_0.weight'] = src['encoder.LayerNorm_0.weight']
    dst['encoder.LayerNorm_0.bias'] = src['encoder.LayerNorm_0.bias']

    # RNN core gating
    for g in ['input_update', 'input_reset', 'state_update', 'state_reset']:
        dst[f'rnn_core.{g}.weight'] = src[f'rnn_core.{g}.weight']
    dst['rnn_core.state_layer_norm.weight'] = src['rnn_core.state_layer_norm.weight']
    dst['rnn_core.state_layer_norm.bias'] = src['rnn_core.state_layer_norm.bias']

    # RNN core transformer blocks
    for i in range(4):
        sp = f'rnn_core.transformer.xa_blocks.{i}'
        dp = f'rnn_core.transformer.xa_blocks.{i}'
        for suffix in [
            'ca_attention_norm.weight', 'ca_attention_norm.bias',
            'ca_attention.query.weight', 'ca_attention.query.bias',
            'ca_attention.key.weight', 'ca_attention.key.bias',
            'ca_attention.value.weight', 'ca_attention.value.bias',
            'ca_attention.out.weight', 'ca_attention.out.bias',
            'mlp_norm.weight', 'mlp_norm.bias',
            'mlp.dense_in.weight', 'mlp.dense_in.bias',
            'mlp.dense_out.weight', 'mlp.dense_out.bias',
            'attention_norm.weight', 'attention_norm.bias',
            'attention.query.weight', 'attention.query.bias',
            'attention.key.weight', 'attention.key.bias',
            'attention.value.weight', 'attention.value.bias',
            'attention.out.weight', 'attention.out.bias',
        ]:
            dst[f'{dp}.{suffix}'] = src[f'{sp}.{suffix}']

    # RNN core output norm
    dst['rnn_core.transformer.output_norm.weight'] = src['rnn_core.transformer.output_norm.weight']
    dst['rnn_core.transformer.output_norm.bias'] = src['rnn_core.transformer.output_norm.bias']

    # Check all keys are assigned
    missing = [k for k in dst if k not in dst or dst[k] is None]
    if missing:
        print(f"WARNING: Missing keys in reference model: {missing}")

    model.load_state_dict(dst, strict=True)
    print("Successfully loaded JAX checkpoint into reference model.")
    return model


# =============================================================================
# Load JAX checkpoint into our implementation
# =============================================================================

def build_our_training_model(wrapper_ckpt_path, img_size=224, num_frames=16):
    """Build the exact model used during training, loaded from the converted checkpoint.

    This uses RVMClassificationWrapper + del_RVMCoreThenReadout from rvm_blocks.py,
    loaded from the _pytorch_wrapper.pth file — exactly as the training script does.
    """
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from models.rvm_modules.rvm_wrapper import RVMViTEncoder, RVMClassificationWrapper
    from models.rvm_modules.rvm_blocks import del_RVMCoreThenReadout
    import torch.nn as nn

    encoder = RVMViTEncoder(
        embed_dim=1024, patch_size=16, num_layers=24,
        num_heads=16, mlp_ratio=4, output_mode="patch_tokens",
        base_token_shape=(16, 16),
    )
    num_patches = (img_size // 16) ** 2 + 1  # +1 for CLS token

    core = del_RVMCoreThenReadout(
        d_model=1024,
        num_rvm_layers=4, num_rvm_heads=16, rvm_mlp_ratio=4, rvm_dropout=0.0,
        num_classes=174,
        readout_num_params=1024, readout_num_queries=1,
        readout_num_heads=16, readout_num_frames=num_frames,
        readout_mlp_ratio=4,
        match_vjepa_implementation=False, add_temporal_posenc=True,
        num_patches=num_patches,
    )

    model = RVMClassificationWrapper(
        encoder=encoder, d_enc=1024,
        sequential_core=core, d_seq_out=174,
        freeze_encoder=True, head=nn.Identity(), pool=None,
    )

    # Load from the SAME converted checkpoint used for training
    model.load_encoder_and_core_weights(wrapper_ckpt_path)
    return model


# =============================================================================
# Main comparison
# =============================================================================

def main():
    img_size = 224
    num_frames = 16  # Same as training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    WRAPPER_CKPT = CHECKPOINT_PATH.replace('.npz', '_pytorch_wrapper.pth')

    # ---- Build reference model (from JAX checkpoint) ----
    print("\n=== Building reference VideoSiamMAE (loaded from JAX .npz) ===")
    base_token_shape = (16, 16)
    ref_model = build_reference_model(base_token_shape=base_token_shape)
    ref_model = load_jax_into_reference(ref_model, CHECKPOINT_PATH)
    ref_model = ref_model.to(device).eval()

    # ---- Build OUR training model (from converted _pytorch_wrapper.pth) ----
    print(f"\n=== Building our training model (loaded from {WRAPPER_CKPT}) ===")
    our_model = build_our_training_model(WRAPPER_CKPT, img_size=img_size, num_frames=num_frames)
    our_model = our_model.to(device).eval()

    # ---- Compare encoder outputs ----
    print(f"\n=== Comparing encoder outputs (img_size={img_size}) ===")
    torch.manual_seed(42)
    x = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        # Reference: tokenize + encode
        import einops
        ref_tokens = ref_model.tokenizer(x)  # (B, h, w, D)
        ref_tokens = einops.rearrange(ref_tokens, 'b h w d -> b (h w) d')
        cls = ref_model.cls_token.expand(1, -1, -1)
        ref_tokens = torch.cat([cls, ref_tokens], dim=1)
        ref_encoded = ref_model.encoder(ref_tokens)  # (B, N+1, D)

        # Ours: encode (using the training model's encoder)
        our_encoded = our_model.encoder(x)  # (B, N+1, D)

    print(f"  Reference encoder output shape: {ref_encoded.shape}")
    print(f"  Our encoder output shape:       {our_encoded.shape}")

    diff = (ref_encoded - our_encoded).abs()
    print(f"  Max abs diff:  {diff.max().item():.2e}")
    print(f"  Mean abs diff: {diff.mean().item():.2e}")

    cls_diff = (ref_encoded[:, 0] - our_encoded[:, 0]).abs()
    print(f"  CLS max diff:  {cls_diff.max().item():.2e}")

    patch_diff = (ref_encoded[:, 1:] - our_encoded[:, 1:]).abs()
    print(f"  Patch max diff: {patch_diff.max().item():.2e}")

    encoder_match = diff.max().item() < 1e-4
    print(f"  Encoder match: {'PASS' if encoder_match else 'FAIL'}")

    # ---- Compare full forward (encoder + core) over num_frames ----
    print(f"\n=== Comparing encoder + core outputs ({num_frames} frames, float32) ===")
    torch.manual_seed(123)
    frames = torch.randn(1, num_frames, 3, img_size, img_size, device=device)

    with torch.no_grad():
        # Reference: process frame by frame
        ref_state = None
        ref_outputs = []
        for t in range(num_frames):
            ref_feat, ref_state = ref_model(frames[:, t], ref_state)
            ref_outputs.append(ref_feat)

        # Ours: process frame by frame through training model's encoder + core
        our_state = None
        our_outputs = []
        for t in range(num_frames):
            our_enc = our_model.encoder(frames[:, t])  # (B, N+1, D)
            if our_state is None:
                our_state = torch.zeros_like(our_enc)
            core_out, our_state = our_model.sequential_core.rvm_core.forward_step(
                our_enc, our_state)
            our_outputs.append(core_out)

    core_match = True
    for t in range(num_frames):
        diff = (ref_outputs[t] - our_outputs[t]).abs()
        max_d = diff.max().item()
        print(f"  Frame {t:2d}: max diff = {max_d:.2e}, "
              f"mean diff = {diff.mean().item():.2e}")
        if max_d > 1e-3:
            core_match = False

    final_diff = (ref_outputs[-1] - our_outputs[-1]).abs().max().item()
    print(f"\n  Core match (float32): {'PASS' if core_match else 'FAIL'} "
          f"(final frame max diff: {final_diff:.2e})")

    # ---- Compare under float16 mixed precision (as used in training) ----
    print(f"\n=== Comparing encoder + core outputs ({num_frames} frames, float16 autocast) ===")
    torch.manual_seed(123)
    frames = torch.randn(1, num_frames, 3, img_size, img_size, device=device)

    with torch.no_grad():
        # Reference in float16
        ref_state_fp16 = None
        ref_outputs_fp16 = []
        for t in range(num_frames):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                ref_feat, ref_state_fp16 = ref_model(frames[:, t], ref_state_fp16)
            ref_outputs_fp16.append(ref_feat.float())
            ref_state_fp16 = ref_state_fp16.float()

        # Ours in float16
        our_state_fp16 = None
        our_outputs_fp16 = []
        for t in range(num_frames):
            with torch.amp.autocast('cuda', dtype=torch.float16):
                our_enc = our_model.encoder(frames[:, t])
                if our_state_fp16 is None:
                    our_state_fp16 = torch.zeros_like(our_enc)
                core_out, our_state_fp16 = our_model.sequential_core.rvm_core.forward_step(
                    our_enc, our_state_fp16)
            our_outputs_fp16.append(core_out.float())
            our_state_fp16 = our_state_fp16.float()

    fp16_core_match = True
    for t in range(num_frames):
        diff = (ref_outputs_fp16[t] - our_outputs_fp16[t]).abs()
        max_d = diff.max().item()
        print(f"  Frame {t:2d}: max diff = {max_d:.2e}, "
              f"mean diff = {diff.mean().item():.2e}")
        if max_d > 1e-1:
            fp16_core_match = False

    fp16_final_diff = (ref_outputs_fp16[-1] - our_outputs_fp16[-1]).abs().max().item()
    print(f"\n  Core match (float16): {'PASS' if fp16_core_match else 'FAIL'} "
          f"(final frame max diff: {fp16_final_diff:.2e})")

    # ---- Also check: float32 ref vs float16 ref (precision degradation) ----
    print(f"\n=== Float32 vs float16 feature degradation (reference model) ===")
    for t in [0, num_frames // 2, num_frames - 1]:
        diff = (ref_outputs[t] - ref_outputs_fp16[t]).abs()
        print(f"  Frame {t:2d}: max diff = {diff.max().item():.2e}, "
              f"mean diff = {diff.mean().item():.2e}")

    # ---- Summary ----
    all_pass = encoder_match and core_match and fp16_core_match
    if all_pass:
        print("\n=== ALL CHECKS PASSED ===")
    else:
        print("\n=== SOME CHECKS FAILED ===")
        if not encoder_match:
            print("  - Encoder outputs differ")
        if not core_match:
            print("  - Core outputs differ (float32)")
        if not fp16_core_match:
            print("  - Core outputs differ (float16)")

    return all_pass


if __name__ == '__main__':
    main()
