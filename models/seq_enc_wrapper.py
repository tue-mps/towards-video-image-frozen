import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


PoolType = Literal["mean", "cls", "mlp"]
ClipStrategy = Literal["causal", "centered", "lookahead"]


class TokenPooler(nn.Module):
    """
    Accepts either:
      - tokens (B, N, D)  -> pools over N
      - features (B, D)   -> pass-through (optionally MLP/projection)
    Always outputs (B, d_out).
    """
    def __init__(self, d_in: int, d_out: int, pool: PoolType = "mean", cls_index: int = 0):
        super().__init__()
        self.pool = pool
        self.d_in = d_in
        self.d_out = d_out

        # For non-MLP paths, we may still need a projection if d_in != d_out
        self.proj = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

        # MLP projects to d_out directly
        self.mlp = None
        if pool == "mlp":
            self.mlp = nn.Sequential(
                nn.LayerNorm(d_in),
                nn.Linear(d_in, d_in),
                nn.GELU(),
                nn.Linear(d_in, d_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"TokenPooler expects (B,D), got {x.shape}"
        # (B, D): already pooled feature
        if self.pool == "mlp":
            return self.mlp(x)                    # (B, d_out)
        else:
            return self.proj(x)                   # (B, d_out), identity if dims match


class SequenceAdapter(nn.Module):
    def __init__(self, core: nn.Module, d_in: int, d_out: int):
        super().__init__()
        self.core = core
        self._supports_parallel = core._supports_parallel
        self._has_step = True  # SequenceAdapter always supports step (enforced below)
        self._state = None

        assert all(hasattr(core, n) for n in ["init_state", "step"]), \
            "SequenceAdapter expects a sequential (step-able) core."

    def init_state(self, B: int, device):
        self._state = self.core.init_state(B, device)

    def reset_state(self, B: Optional[int] = None, device: Optional[torch.device] = None):
        self._state = None
        if B is not None and device is not None:
            self._state = self.core.init_state(B, device)

    def detach_state(self):
        """Detach any tensor(s) inside the recurrent state in-place (TBPTT boundary)."""
        def _detach(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            if isinstance(obj, dict):
                return {k: _detach(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_detach(v) for v in obj)
            return obj
        if self._state is not None:
            self._state = _detach(self._state)

    def _maybe_reinit(self, x: torch.Tensor):
        B, dev = x.size(0), x.device
        need = self._state is None or (isinstance(self._state, torch.Tensor) and self._state.size(0) != B)
        if need:
            self._state = self.core.init_state(B, dev)

    def forward(self, x_seq: torch.Tensor, *, reset_state: bool = True, reset_mask: Optional[torch.Tensor] = None, prefer_parallel: bool = True, **kwargs) -> torch.Tensor:
        if prefer_parallel and self._supports_parallel:
            return self.core(x_seq, **kwargs)

        B, T = x_seq.size(0), x_seq.size(1)
        if reset_state or self._state is None:
            self.init_state(B, x_seq.device)

        if reset_mask is not None:
            # only correct if your core implements masked_reset
            if hasattr(self.core, "masked_reset"):
                self._state = self.core.masked_reset(self._state, reset_mask)
            else:
                # if you truly rely on reset_mask, you probably want to require masked_reset
                raise RuntimeError("reset_mask provided but core has no masked_reset().")

        ys = []
        for t in range(T):
            y_t, self._state = self.core.step(x_seq[:, t], self._state)
            ys.append(y_t)
        return torch.stack(ys, dim=1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, *, reset: bool = False, reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._maybe_reinit(x_t)
        if reset:
            self.init_state(x_t.size(0), x_t.device)
        elif reset_mask is not None:
            if hasattr(self.core, "masked_reset"):
                self._state = self.core.masked_reset(self._state, reset_mask)
            else:
                raise RuntimeError("reset_mask provided but core has no masked_reset().")

        y_t, self._state = self.core.step(x_t, self._state)
        return y_t


class ImageSequentialWrapper1F(nn.Module):
    """
    Sequential wrapper for single-frame (image) encoders like DINOv2.

    This is a simplified version of VideoSequentialWrapper2F that:
    - Takes single frames (B, C, H, W) instead of 2-frame clips
    - Does not need frame buffering for streaming inference
    - Uses enc_indices=[0] (single frame per timestep)

    Inputs:
      - Training:  x_seq  : (B, T, C, H, W)
      - Inference: frame_t: (B, C, H, W) via step(...)

    Pipeline:
      - encoder: (B, C, H, W) -> (B, N, D_enc) or (B, D_enc)
      - pool -> (B, d_seq_in)
      - seq  -> (B, d_seq_out)
      - head -> (B, out_dim)
    """
    def __init__(
        self,
        encoder: nn.Module,         # expects (B, C, H, W) -> (B, N, D_enc) or (B, D_enc)
        d_enc: int,
        d_seq_in: int,
        sequential_core: nn.Module, # step-able preferred
        d_seq_out: int,
        freeze_encoder: bool = True,
        head: Optional[nn.Module] = None,
        pool: PoolType = "mean",
        cls_index: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_clip_len = 1  # always 1 for image encoder
        if pool is None:
            self.pooler = nn.Identity()
        else:
            self.pooler = TokenPooler(d_in=d_enc, d_out=d_seq_in, pool=pool, cls_index=cls_index)
        self.seq = SequenceAdapter(sequential_core, d_in=d_seq_in, d_out=d_seq_out)
        self.head = head if head is not None else nn.Identity()

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
        return self

    def get_train_params(self):
        """All parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    def get_train_param_groups(
        self,
        weight_decay: float = 0.05,
        separate_head: bool = True,
    ):
        """
        Returns optimizer param groups:
          - decay:      weights (matrix/tensor params)
          - no_decay:   biases and norm scales (LayerNorm/BatchNorm/etc.)
          - optionally split head params into their own groups (useful for a higher LR)
        """
        decay, no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = name.startswith("head.")

            # heuristic: no weight decay for biases and 1D params (norm scales)
            nd = (p.ndim <= 1) or name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower()) or ("ln" in name.lower())

            if separate_head and is_head:
                (head_no_decay if nd else head_decay).append(p)
            else:
                (no_decay if nd else decay).append(p)

        groups = []
        if decay:        groups.append({"params": decay,        "weight_decay": weight_decay})
        if no_decay:     groups.append({"params": no_decay,     "weight_decay": 0.0})
        if head_decay:   groups.append({"params": head_decay,   "weight_decay": weight_decay})
        if head_no_decay:groups.append({"params": head_no_decay,"weight_decay": 0.0})
        return groups

    def forward_sequence_framewise(
        self,
        x_seq: torch.Tensor,
        enc_indices: list[int],  # should be [0] for image encoder
        *,
        reset_state: bool = True,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Train-time path: encode and step the recurrent core frame-by-frame.

        x_seq: (B, T, C, H, W)
        enc_indices: should be [0] for single-frame encoder
        returns: (B, T, out_dim)
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Init/reset recurrent state once at the sequence start
        if self.seq._has_step:
            if reset_state or self.seq._state is None:
                self.seq.init_state(B, device)
            if reset_mask is not None:
                self.seq.reset_state(B, device)
                if hasattr(self.seq.core, "masked_reset"):
                    self.seq._state = self.seq.core.masked_reset(self.seq._state, reset_mask)

        ys = []
        for t in range(T):
            frame = x_seq[:, t]                             # (B, C, H, W)
            tokens = self.encoder(frame)                    # (B, N, D_enc) or (B, D_enc)
            feat = self.pooler(tokens)                      # (B, d_seq_in)

            if self.seq._has_step:
                y_t, self.seq._state = self.seq.core.step(feat, self.seq._state)
            else:
                raise RuntimeError("Sequential core must be step-able for framewise training.")
            ys.append(y_t)

        y = torch.stack(ys, dim=1)       # (B, T, d_seq_out)
        return self.head(y)              # (B, T, out_dim)

    def forward_sequence_batched(
        self,
        x_seq: torch.Tensor,
        enc_indices: list[int],  # should be [0] for image encoder
        *,
        reset_state: bool = True,
        reset_mask: Optional[torch.Tensor] = None,
        encoder_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Efficient train-time path: batch-encode ALL frames at once, then step the recurrent core.

        x_seq: (B, T, C, H, W)
        enc_indices: should be [0] for single-frame encoder
        encoder_batch_size: if set, encode frames in chunks of this size to limit GPU memory
        returns: (B, T, out_dim)
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # ========== STEP 1: Batch-encode all frames ==========
        frames_flat = x_seq.view(B * T, C, H, W).contiguous()

        if encoder_batch_size is None or B * T <= encoder_batch_size:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                tokens = self.encoder(frames_flat)  # (B*T, N, D_enc) or (B*T, D_enc)
            feats_flat = self.pooler(tokens)        # (B*T, d_seq_in)
        else:
            feats_chunks = []
            for i in range(0, B * T, encoder_batch_size):
                chunk = frames_flat[i:i + encoder_batch_size]
                with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                    tokens_chunk = self.encoder(chunk)
                feats_chunk = self.pooler(tokens_chunk)
                feats_chunks.append(feats_chunk)
            feats_flat = torch.cat(feats_chunks, dim=0)

        # Reshape to (B, T, ...) handling both:
        # - (B*T, D)        -> (B, T, D)         when pool reduces tokens
        # - (B*T, N, D_enc) -> (B, T, N, D_enc)  when pool=None (MambaThenPool handles pooling)
        if feats_flat.dim() == 2:
            feats = feats_flat.view(B, T, feats_flat.shape[1]).contiguous()
        elif feats_flat.dim() == 3:
            feats = feats_flat.view(B, T, feats_flat.shape[1], feats_flat.shape[2]).contiguous()
        else:
            raise ValueError(f"Unexpected feats_flat.dim(): {feats_flat.dim()}")

        # ========== STEP 2: Sequential core (in parallel or in a loop over T) ==========
        if self.seq._supports_parallel:
            y = self.seq(feats, reset_state=reset_state, reset_mask=reset_mask,
                     prefer_parallel=True, return_sequence=True)  # (B, T, d_seq_out)
        else:
            if reset_state or self.seq._state is None:
                self.seq.init_state(B, device)
            if reset_mask is not None:
                self.seq.reset_state(B, device)
                if hasattr(self.seq.core, "masked_reset"):
                    self.seq._state = self.seq.core.masked_reset(self.seq._state, reset_mask)

            ys = []
            for t in range(T):
                y_t, self.seq._state = self.seq.core.step(feats[:, t], self.seq._state)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)  # (B, T, d_seq_out)

        return self.head(y)  # (B, T, out_dim)

    def forward(self, x_seq: torch.Tensor, *, enc_indices: list[int], reset_state: bool = True, reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_sequence_framewise(x_seq, enc_indices=enc_indices, reset_state=reset_state)

    # ---------- STREAMING INFERENCE ----------
    @torch.no_grad()
    def step(self, frame_t: torch.Tensor, *, reset: bool = False, reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frame_t: (B, C, H, W) — single frame at time t
        """
        tokens = self.encoder(frame_t)      # (B, N, D_enc) or (B, D_enc)
        feat = self.pooler(tokens)          # (B, d_seq_in)
        y_t = self.seq.step(feat, reset=reset, reset_mask=reset_mask)  # (B, d_seq_out)
        return self.head(y_t)

    def reset_state(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        self.seq.reset_state(B=batch_size, device=device)


class VideoSequentialWrapper2F(nn.Module):
    """
    Inputs:
      - Training:  x_seq  : (B, T, C, H, W)
      - Inference: frame_t: (B, C, H, W) via step(...)

    Internals:
      - Assembles 2-frame clips per step according to `clip_strategy`.
      - encoder: (B, C, 2, H, W) -> (B, N, D_enc)
      - pool -> (B, d_seq_in)
      - seq  -> (B, d_seq_out)
      - head -> (B, out_dim)
    """
    def __init__(
        self,
        encoder: nn.Module,         # expects (B, C, 2, H, W) -> (B,N,D_enc)
        encoder_clip_len: int,  # must be 1 or 2 for this module
        d_enc: int,
        d_seq_in: int,
        sequential_core: nn.Module, # step-able preferred
        d_seq_out: int,
        freeze_encoder: bool = True,
        head: Optional[nn.Module] = None,
        pool: PoolType = "mean",
        cls_index: int = 0,
        clip_strategy: ClipStrategy = "causal",  # 'causal' uses (t-1, t)
        pad_mode: Literal["replicate", "zero"] = "replicate",
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_clip_len = encoder_clip_len  # 1 = image encoder, 2 = 2-frame encoder
        if pool is None:
            self.pooler = nn.Identity()
        else:
            self.pooler = TokenPooler(d_in=d_enc, d_out=d_seq_in, pool=pool, cls_index=cls_index)
        self.seq = SequenceAdapter(sequential_core, d_in=d_seq_in, d_out=d_seq_out)
        self.head = head if head is not None else nn.Identity()
        self.clip_strategy = clip_strategy
        self.pad_mode = pad_mode

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        # tiny frame buffer for streaming (stores last 1 frame for 2-frame clips)
        self._frame_buffer: Optional[torch.Tensor] = None  # (B, C, H, W)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            # keep only the encoder in eval; rest of the model stays in 'mode'
            self.encoder.eval()
        return self
    
    def get_train_params(self):
        """All parameters with requires_grad=True."""
        return (p for p in self.parameters() if p.requires_grad)

    def get_train_param_groups(
        self,
        weight_decay: float = 0.05,
        separate_head: bool = True,
    ):
        """
        Returns optimizer param groups:
          - decay:      weights (matrix/tensor params)
          - no_decay:   biases and norm scales (LayerNorm/BatchNorm/etc.)
          - optionally split head params into their own groups (useful for a higher LR)
        """
        decay, no_decay = [], []
        head_decay, head_no_decay = [], []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = name.startswith("head.")

            # heuristic: no weight decay for biases and 1D params (norm scales)
            nd = (p.ndim <= 1) or name.endswith(".bias") or ("norm" in name.lower()) or ("bn" in name.lower()) or ("ln" in name.lower())

            if separate_head and is_head:
                (head_no_decay if nd else head_decay).append(p)
            else:
                (no_decay if nd else decay).append(p)

        groups = []
        if decay:        groups.append({"params": decay,        "weight_decay": weight_decay})
        if no_decay:     groups.append({"params": no_decay,     "weight_decay": 0.0})
        if head_decay:   groups.append({"params": head_decay,   "weight_decay": weight_decay})
        if head_no_decay:groups.append({"params": head_no_decay,"weight_decay": 0.0})
        return groups    

    # ---------- clip assembly helpers ----------
    def _pair_indices(self, T: int):
        if self.clip_strategy == "causal":
            # (t-1, t); for t=0, use (0,0) and rely on pad_mode
            left = [max(0, t-1) for t in range(T)]
            right = list(range(T))
        elif self.clip_strategy == "lookahead":
            # (t, t+1); last step repeats last frame
            left = list(range(T))
            right = [min(T-1, t+1) for t in range(T)]
        elif self.clip_strategy == "centered":
            # round down center: (t, t+1) except t=T-1 -> (T-2, T-1)
            left = [min(t, T-2) for t in range(T)]
            right = [min(t+1, T-1) for t in range(T)]
        else:
            raise ValueError(self.clip_strategy)
        return left, right

    def _build_2f_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B,T,C,H,W) -> clips: (B,T,C,2,H,W)
        """
        B, T, C, H, W = x_seq.shape
        L, R = self._pair_indices(T)
        left = x_seq[:, torch.tensor(L, device=x_seq.device), ...]   # (B,T,C,H,W)
        right = x_seq[:, torch.tensor(R, device=x_seq.device), ...]  # (B,T,C,H,W)

        if self.clip_strategy == "causal" and self.pad_mode == "replicate":
            # ensure true replication at t=0: (0,0)
            left[:, 0] = x_seq[:, 0]

        clips = torch.stack([left, right], dim=3)  # (B,T,C,2,H,W)
        return clips
    
    # def _encode_chunk(self, x_seq: torch.Tensor, t0: int, t1: int) -> torch.Tensor:
    #     """
    #     Slice [t0, t1) and build causal pairs (t-1, t) for that window only.
    #     Returns (B, L, d_seq_in) where L = t1 - t0.
    #     """
    #     B, T, C, H, W = x_seq.shape
    #     L = t1 - t0
    #     idx = torch.arange(t0, t1, device=x_seq.device)                 # (L,)
    #     right = x_seq[:, idx, ...]                                      # (B,L,C,H,W)
    #     left_idx = torch.clamp(idx - 1, min=0)                          # (L,) replicate at 0
    #     left = x_seq[:, left_idx, ...]                                  # (B,L,C,H,W)
    #     clips = torch.stack([left, right], dim=3)                        # (B,L,C,2,H,W)
    #     flat = clips.view(B * L, C, 2, H, W)
    #     tokens = self.encoder(flat)                                      # (B*L, N, D_enc) or (B*L, D_enc)
    #     feats = self.pooler(tokens)                                      # (B*L, d_seq_in)
    #     return feats.view(B, L, -1)                                      # (B,L,d_seq_in)

    # def _encode_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
    #     """
    #     x_seq: (B,T,C,H,W) -> (B,T,d_seq_in)
    #     """
    #     B, T, C, H, W = x_seq.shape
    #     clips = self._build_2f_sequence(x_seq)              # (B,T,C,2,H,W)
    #     flat = clips.view(B*T, C, 2, H, W)
    #     tokens = self.encoder(flat)                         # (B*T,N,D_enc)
    #     print(f"tokens: {tokens.shape}")
    #     pooled = self.pooler(tokens)                        # (B*T,d_seq_in)
    #     print(f"pooled: {pooled.shape}")
    #     return pooled.view(B, T, -1)

    def _encode_step_from_frames(self, prev_frame: torch.Tensor, cur_frame: torch.Tensor) -> torch.Tensor:
        """
        prev_frame, cur_frame: (B,C,H,W) -> feature (B,d_seq_in)
        """
        clips = torch.stack([prev_frame, cur_frame], dim=2)  # (B,C,2,H,W)
        tokens = self.encoder(clips)                         # (B,N,D_enc)
        feat = self.pooler(tokens)                           # (B,d_seq_in)
        return feat

    def forward_sequence_framewise(
        self,
        x_seq: torch.Tensor,
        enc_indices: list[int],  # indices of the frames to pass to encoder (because of encoder_fps / seq_fps ratio)
        *,
        reset_state: bool = True,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Train-time, memory-friendly path: encode and step the recurrent core frame-by-frame.
        
        For 2-frame encoder (enc_frames=2):
          - Input x_seq has T+(enc_frames-1) frames where the first (enc_frames-1) frames 
            are solely context for the encoder. Encoder takes enc_frames=2. 
          - We form pairs: (t-1, t) for t in [1, T], giving T encoder outputs
          - Returns (B, T, out_dim) - predictions for frames (enc_frames-1)=1 to T
          
        For 1-frame encoder (enc_frames=1):
          - Input x_seq has shape (B, T, C, H, W)
          - All T frames are processed
          - Returns (B, T, out_dim)
        
        x_seq: (B, T_input, C, H, W) where T_input = T + (enc_frames - 1)
        returns: (B, T, out_dim)

        enc_indices defines the frame offsets for the encoder at timestep 0
        e.g., [0, 1] means encoder sees frames 0,1 then 1,2 then 2,3 ...
        e.g., [0, 3] means encoder sees frames 0,3 then 1,4 then 2,5 ...
        """
        B, T_input, C, H, W = x_seq.shape
        device = x_seq.device
        T = T_input - enc_indices[-1]

        # Init/reset recurrent state once at the sequence start
        if self.seq._has_step:
            if reset_state or self.seq._state is None:
                self.seq.init_state(B, device)
            if reset_mask is not None:
                # one-shot masked reset at sequence start if requested
                self.seq.reset_state(B, device)
                if hasattr(self.seq.core, "masked_reset"):
                    self.seq._state = self.seq.core.masked_reset(self.seq._state, reset_mask)

        ys = []

        if self.encoder_clip_len == 2:
            # For each output timestep t, shift enc_indices by t
            # t=0: frames [enc_indices[0], enc_indices[1]] = [0, 3]
            # t=1: frames [enc_indices[0]+1, enc_indices[1]+1] = [1, 4]
            # t=2: frames [enc_indices[0]+2, enc_indices[1]+2] = [2, 5]
            for t in range(T):
                prev_idx = enc_indices[0] + t
                cur_idx = enc_indices[1] + t
                
                prev = x_seq[:, prev_idx]  # (B, C, H, W)
                cur = x_seq[:, cur_idx]    # (B, C, H, W)
                clips = torch.stack([prev, cur], dim=2)  # (B, C, 2, H, W)
                tokens = self.encoder(clips)             # (B, N, D_enc) or (B, D_enc)
                feat = self.pooler(tokens)               # (B, d_seq_in)

                # Recurrent step with grads
                if self.seq._has_step:
                    y_t, self.seq._state = self.seq.core.step(feat, self.seq._state)
                else:
                    raise RuntimeError("Sequential core must be step-able for framewise training.")
                ys.append(y_t)

        elif self.encoder_clip_len == 1:
            for t in range(T):
                frame = x_seq[:, t]                             # (B,C,H,W)
                tokens = self.encoder(frame)                    # (B,N,D_enc) or (B,D_enc)
                feat = self.pooler(tokens)                      # (B,d_seq_in)

        y_t, self.seq._state = self.seq.core.step(feat, self.seq._state)
        ys.append(y_t)

        y = torch.stack(ys, dim=1)       # (B, T, d_seq_out)
        return self.head(y)              # (B, T, out_dim)

    def forward_sequence_batched(
        self,
        x_seq: torch.Tensor,
        enc_indices: list[int],  # indices of the frames to pass to encoder (because of encoder_fps / seq_fps ratio)
        *,
        reset_state: bool = True,
        reset_mask: Optional[torch.Tensor] = None,
        encoder_batch_size: Optional[int] = None,  # if set, encode in chunks to limit memory
    ) -> torch.Tensor:
        """
        Efficient train-time path: batch-encode ALL clips at once, then step the recurrent core.
        
        This is much faster than forward_sequence_framewise because the encoder is called
        only once (or a few times if encoder_batch_size is set) instead of T times.
        
        For 2-frame encoder (enc_frames=2):
          - Input x_seq has T+(enc_indices[-1]) frames
          - We form all T pairs at once and encode in a single batch
          - Then loop only for the recurrent sequential core
          - Returns (B, T, out_dim)
        
        x_seq: (B, T_input, C, H, W) where T_input = T + enc_indices[-1]
        enc_indices: frame offsets for the encoder at timestep 0
            e.g., [0, 1] means encoder sees frames 0,1 then 1,2 then 2,3 ...
            e.g., [0, 3] means encoder sees frames 0,3 then 1,4 then 2,5 ...
        encoder_batch_size: if set, encode clips in chunks of this size to limit GPU memory
            e.g., encoder_batch_size=512 encodes 512 clips at a time
        
        returns: (B, T, out_dim)
        """
        B, T_input, C, H, W = x_seq.shape
        device = x_seq.device
        T = T_input - enc_indices[-1]

        # ========== STEP 1: Build all 2-frame clips and encode in batch ==========
        if self.encoder_clip_len == 2:
            # Build index tensors for all T timesteps at once
            t_indices = torch.arange(T, device=device)  # (T,)
            prev_indices = enc_indices[0] + t_indices   # (T,)
            cur_indices = enc_indices[1] + t_indices    # (T,)
            
            # Gather all frames: (B, T, C, H, W) for prev and cur
            prev_frames = x_seq[:, prev_indices]  # (B, T, C, H, W)
            cur_frames = x_seq[:, cur_indices]    # (B, T, C, H, W)
            
            # Stack to form clips: (B, T, C, 2, H, W)
            clips = torch.stack([prev_frames, cur_frames], dim=3)  # (B, T, C, 2, H, W)
            
            # Flatten batch and time: (B*T, C, 2, H, W)
            clips_flat = clips.view(B * T, C, 2, H, W).contiguous()
            
            # Encode all clips (optionally in chunks to limit memory)
            if encoder_batch_size is None or B * T <= encoder_batch_size:
                # Single batch encode
                with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                    tokens = self.encoder(clips_flat)  # (B*T, N, D_enc) or (B*T, D_enc)
                feats_flat = self.pooler(tokens)       # (B*T, d_seq_in) or (B*T, N, d_seq_in)
            else:
                # Chunked encode to limit memory
                feats_chunks = []
                for i in range(0, B * T, encoder_batch_size):
                    chunk = clips_flat[i:i + encoder_batch_size]
                    with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                        tokens_chunk = self.encoder(chunk)
                    feats_chunk = self.pooler(tokens_chunk)
                    feats_chunks.append(feats_chunk)
                feats_flat = torch.cat(feats_chunks, dim=0)  # (B*T, d_seq_in)
            
            # Reshape to (B, T, ...) handling both:
            # - (B*T, D)           -> (B, T, D)
            # - (B*T, N, D_enc)    -> (B, T, N, D_enc)
            if feats_flat.dim() == 2:
                assert feats_flat.shape[0] == B * T
                feats = feats_flat.view(B, T, feats_flat.shape[1]).contiguous()
            elif feats_flat.dim() == 3:
                assert feats_flat.shape[0] == B * T
                feats = feats_flat.view(B, T, feats_flat.shape[1], feats_flat.shape[2]).contiguous()
            else:
                raise ValueError(f"Unexpected feats_flat.dim(): {feats_flat.dim()}")
        
        elif self.encoder_clip_len == 1:
            # Single-frame encoder: just encode all T frames
            frames_flat = x_seq[:, :T].reshape(B * T, C, H, W).contiguous()
            
            if encoder_batch_size is None or B * T <= encoder_batch_size:
                with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                    tokens = self.encoder(frames_flat)
                feats_flat = self.pooler(tokens)
            else:
                feats_chunks = []
                for i in range(0, B * T, encoder_batch_size):
                    chunk = frames_flat[i:i + encoder_batch_size]
                    with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                        tokens_chunk = self.encoder(chunk)
                    feats_chunk = self.pooler(tokens_chunk)
                    feats_chunks.append(feats_chunk)
                feats_flat = torch.cat(feats_chunks, dim=0)
            
            if feats_flat.dim() == 2:
                assert feats_flat.shape[0] == B * T
                feats = feats_flat.view(B, T, feats_flat.shape[1]).contiguous()
            elif feats_flat.dim() == 3:
                assert feats_flat.shape[0] == B * T
                feats = feats_flat.view(B, T, feats_flat.shape[1], feats_flat.shape[2]).contiguous()
            else:
                raise ValueError(f"Unexpected feats_flat.dim(): {feats_flat.dim()}")
        else:
            raise ValueError(f"encoder_clip_len must be 1 or 2, got {self.encoder_clip_len}")

        # ========== STEP 2: Sequential core (in parallel or in a loop over T) ==========
        if self.seq._supports_parallel:
            y = self.seq(feats, reset_state=reset_state, reset_mask=reset_mask,
                     prefer_parallel=True, return_sequence=True)  # (B, T, d_seq_out)
        else:
            if reset_state or self.seq._state is None:
                self.seq.init_state(B, device)
            if reset_mask is not None:
                self.seq.reset_state(B, device)
                if hasattr(self.seq.core, "masked_reset"):
                    self.seq._state = self.seq.core.masked_reset(self.seq._state, reset_mask)
            
            ys = []
            for t in range(T):
                y_t, self.seq._state = self.seq.core.step(feats[:, t], self.seq._state)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)  # (B, T, d_seq_out)

        return self.head(y)  # (B, T, out_dim)

    def forward(self, x_seq: torch.Tensor, *, enc_indices: list[int], reset_state: bool = True, reset_mask: Optional[torch.Tensor] = None, detach_between_chunks: bool = True) -> torch.Tensor:
        return self.forward_sequence_framewise(x_seq, enc_indices=enc_indices, reset_state=reset_state)

    # ---------- STREAMING INFERENCE ----------
    @torch.no_grad()
    def step(self, frame_t: torch.Tensor, *, reset: bool = False, reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        frame_t: (B,C,H,W)  — one frame at time t
        - keeps an internal 1-frame buffer to build (t-1, t) for the encoder
        """
        B, C, H, W = frame_t.shape
        dev = frame_t.device

        # handle frame buffer
        if reset or self._frame_buffer is None or (self._frame_buffer.size(0) != B):
            # initialize buffer: replicate current frame (causal start)
            self._frame_buffer = frame_t.clone()

        if reset_mask is not None:
            # per-sample buffer reset
            mask = reset_mask.to(torch.bool)
            self._frame_buffer[mask] = frame_t[mask]  # start pairs as (t,t) for those samples

        # build 2-frame clip: (prev, cur)
        prev = self._frame_buffer
        cur = frame_t
        feat = self._encode_step_from_frames(prev, cur)          # (B,d_seq_in)

        # advance buffer
        self._frame_buffer = frame_t.clone()

        y_t = self.seq.step(feat, reset=reset, reset_mask=reset_mask)  # (B,d_seq_out)
        return self.head(y_t)

    def reset_state(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        self.seq.reset_state(B=batch_size, device=device)  # <-- not B=
        self._frame_buffer = None


# TODO: specific to encoder
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

    return missing_keys


