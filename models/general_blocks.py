import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable


class AttentionPooling(nn.Module):
    """
    Learnable probe token(s) attend over patch tokens to produce pooled representation(s).

    Input:  x (B, N, D)  patch tokens
    Output: y (B, P, D)  pooled probe tokens (P=num_probe)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_probe = num_probe

        self.probe = nn.Parameter(torch.randn(1, num_probe, embed_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm1_q = norm_layer(embed_dim)
        self.norm1_kv = norm_layer(embed_dim)

        mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(embed_dim, mlp_width)),
                    ("act", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, embed_dim)),
                    ("drop", nn.Dropout(proj_dropout)),
                ]
            )
        )
        self.norm2 = norm_layer(embed_dim)
        self.drop_path = nn.Dropout(proj_dropout)  # simple residual dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: (B, P, D)
        """
        B, N, D = x.shape
        assert D == self.embed_dim, f"Expected D={self.embed_dim}, got {D}"

        # Expand learnable probes for this batch
        q = self.probe.expand(B, -1, -1)  # (B, P, D)

        # Cross-attention: probes query the patch tokens
        qn = self.norm1_q(q)
        xk = self.norm1_kv(x)

        attn_out, _ = self.attn(query=qn, key=xk, value=xk, need_weights=False)
        q = q + self.drop_path(attn_out)

        # MLP on probes
        q = q + self.mlp(self.norm2(q))
        return q


class MeanPooling(nn.Module):
    """
    Simple mean pooling with optional projection.
    Much faster than attention pooling, suitable for experiments.
    
    Input:  x (B, N, D) patch tokens
    Output: y (B, D) or (B, 1, D) pooled representation
    """
    def __init__(
        self,
        embed_dim: int,
        proj_dim: int = None,
        num_probe: int = 1,  # ignored, for API compatibility
        **kwargs  # ignore other args
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_probe = 1  # always 1 for mean pooling
        
        if proj_dim is not None and proj_dim != embed_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, proj_dim),
            )
            self.out_dim = proj_dim
        else:
            self.proj = nn.LayerNorm(embed_dim)
            self.out_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: (B, 1, D) for API compatibility with AttentionPooling
        """
        pooled = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        return self.proj(pooled)


class ConvPooling(nn.Module):
    """
    1D convolution-based pooling over tokens.
    Faster than attention, learns local patterns.
    
    Input:  x (B, N, D) patch tokens
    Output: y (B, 1, D) pooled representation
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,  # ignored, for API compatibility
        num_probe: int = 1,  # ignored
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_probe = 1
        
        # Norm before conv
        self.norm_in = nn.LayerNorm(embed_dim)
        
        # Conv stack to aggregate tokens (operates on transposed input)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        
        # MLP for further processing
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        
        self.norm_out = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        returns: (B, 1, D) for API compatibility
        """
        # Apply LayerNorm before transpose
        x = self.norm_in(x)  # (B, N, D)
        
        # Conv expects (B, D, N)
        x_t = x.transpose(1, 2)  # (B, D, N)
        x_t = self.conv(x_t)     # (B, D, N)
        
        # Global average pool + MLP
        pooled = x_t.mean(dim=2)  # (B, D)
        pooled = pooled + self.mlp(pooled)  # residual MLP
        pooled = self.norm_out(pooled)
        
        return pooled.unsqueeze(1)  # (B, 1, D)
