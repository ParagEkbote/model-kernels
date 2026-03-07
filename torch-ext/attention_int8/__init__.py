from typing import Optional

import torch

from . import int8_attention as _ops


def int8_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    timestep_scales: Optional[torch.Tensor] = None,
    timestep: int = 0,
    causal: bool = False,
) -> torch.Tensor:
    """INT8 fused attention forward pass.
    
    Args:
        Q: Query tensor [B, H, N, D] in float16
        K: Key tensor [B, kv_H, N, D] in float16
        V: Value tensor [B, kv_H, N, D] in float16
        timestep_scales: Optional timestep scales [num_timesteps] in float32
        timestep: Current timestep index (default: 0)
        causal: Apply causal masking (default: False)
    
    Returns:
        Output tensor [B, H, N, D] in float16
    """
    return _ops.int8_attention_forward(Q, K, V, timestep_scales, timestep, causal)


def is_head_dim_supported(D: int) -> bool:
    """Check if HEAD_DIM is supported."""
    return _ops.is_head_dim_supported(D)


def get_supported_head_dims() -> list:
    """Get list of supported HEAD_DIM values."""
    return _ops.get_supported_head_dims()


def get_block_config(D: int) -> tuple:
    """Get block tile configuration (BQ, BK) for HEAD_DIM."""
    return _ops.get_block_config(D)


def get_int8_attention_smem_bytes(D: int) -> int:
    """Get shared memory requirement in bytes for HEAD_DIM."""
    return _ops.get_int8_attention_smem_bytes(D)


def get_occupancy_hint(D: int) -> int:
    """Get occupancy hint for HEAD_DIM."""
    return _ops.get_occupancy_hint(D)


__all__ = [
    "int8_attention_forward",
    "is_head_dim_supported",
    "get_supported_head_dims",
    "get_block_config",
    "get_int8_attention_smem_bytes",
    "get_occupancy_hint",
]