"""RVM (Recurrent Video Masked Autoencoders) modules.

Pure PyTorch implementation of RVM, plus task-specific wrappers built on
top of a frozen image encoder + recurrent temporal core + attentive readout.

Reference: Zoran et al., 2025
(https://github.com/google-deepmind/representations4d).

Mamba-based recurrent cores live in models/ssm_modules/.
"""

from .rvm_blocks import (
    TransformerMLP,
    RVMReadout,
    RVMCrossAttentionBlock,
    RVMCrossAttentionTransformer,
    IdentityCore,
    GatedTransformerCore,
)

from .rvm_wrapper import (
    # RVM encoder
    build_rvm_encoder,
    RVMViTEncoder,
    # Classification
    RVMClassificationWrapper,
    RVMClassificationOnlyReadout,
    RVMStreamingClassificationWrapper,
    # Tracking
    FourierPositionEncoding,
    BBoxQueryEncoder,
    TrackingCrossAttentionReadout,
    RVMTrackingWrapper,
    RVMStreamingTrackingWrapper,
    # Depth (streaming-only)
    RVMDepthOnlyReadout,
    RVMStreamingDepthWrapper,
    # Point Tracking
    PointQueryEncoder4DS,
    PointTrackingReadout4DS,
    RVMPointTrackingOnlyReadout,
    RVMPointTrackingWrapper,
    RVMStreamingPointTrackingWrapper,
    # Camera Pose (streaming-only)
    RVMStreamingCameraPoseWrapper,
)

__all__ = [
    # Blocks
    "TransformerMLP",
    "RVMReadout",
    "RVMCrossAttentionBlock",
    "RVMCrossAttentionTransformer",
    "IdentityCore",
    "GatedTransformerCore",
    # RVM encoder
    "build_rvm_encoder",
    "RVMViTEncoder",
    # Classification
    "RVMClassificationWrapper",
    "RVMClassificationOnlyReadout",
    "RVMStreamingClassificationWrapper",
    # Tracking
    "FourierPositionEncoding",
    "BBoxQueryEncoder",
    "TrackingCrossAttentionReadout",
    "RVMTrackingWrapper",
    "RVMStreamingTrackingWrapper",
    # Depth (streaming-only)
    "RVMDepthOnlyReadout",
    "RVMStreamingDepthWrapper",
    # Point Tracking
    "PointQueryEncoder4DS",
    "PointTrackingReadout4DS",
    "RVMPointTrackingOnlyReadout",
    "RVMPointTrackingWrapper",
    "RVMStreamingPointTrackingWrapper",
    # Camera Pose (streaming-only)
    "RVMStreamingCameraPoseWrapper",
]
