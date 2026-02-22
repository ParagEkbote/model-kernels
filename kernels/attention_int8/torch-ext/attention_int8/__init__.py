from torch.utils.cpp_extension import load
import os
import torch

# -------------------------------------------------------------------------
# Import compiled extension
# -------------------------------------------------------------------------

try:
    from . import _ops as ops
except ImportError:
    ops = None

# -------------------------------------------------------------------------
# Register custom op (torch.compile compatible)
# -------------------------------------------------------------------------

if ops is not None:
    try:
        @torch.library.custom_op(
            "kernels::int8_attention_forward",
            mutates_args=()
        )
        def _int8_attention_custom_op(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            timestep_scales: torch.Tensor | None = None,
            timestep: int = 0,
            causal: bool = False,
        ) -> torch.Tensor:
            return ops.int8_attention_forward(
                Q.contiguous(),
                K.contiguous(),
                V.contiguous(),
                timestep_scales.contiguous() if timestep_scales is not None else None,
                int(timestep),
                bool(causal),
            )

        @_int8_attention_custom_op.register_fake
        def _int8_attention_fake(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            timestep_scales: torch.Tensor | None = None,
            timestep: int = 0,
            causal: bool = False,
        ) -> torch.Tensor:
            return Q.new_empty(Q.shape)

        _CUSTOM_OP_REGISTERED = True

    except (AttributeError, Exception):
        # torch.library.custom_op not available
        pass

# -------------------------------------------------------------------------
# JIT build fallback (development mode)
# -------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(__file__)

_SRC = [
    os.path.join(_THIS_DIR, "attention_int8.cu"),
    os.path.join(_THIS_DIR, "torch_binding.cpp"),
]

def _build_module():
    return load(
        name="attention_int8_ext",
        sources=_SRC,
        verbose=False,
    )

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        _mod = _build_module()
    return _mod

# -------------------------------------------------------------------------
# Public Python wrapper
# -------------------------------------------------------------------------

def int8_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    timestep_scales: torch.Tensor | None = None,
    timestep: int = 0,
    causal: bool = False,
) -> torch.Tensor:
    """
    Run the fused INT8 diffusion attention kernel.

    Args:
        Q: Tensor [B, H, N, D], float16, CUDA
        K: Tensor [B, kv_H, N, D], float16, CUDA
        V: Tensor [B, kv_H, N, D], float16, CUDA
        timestep_scales: Optional float32 CUDA tensor [T]
        timestep: integer timestep index
        causal: whether to apply causal masking

    Returns:
        Tensor [B, H, N, D], float16
    """

    if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
        raise ValueError("Q, K, V must be CUDA tensors")

    if Q.dtype != torch.float16:
        raise ValueError("Q must be float16")

    if Q.shape[-1] % 16 != 0:
        raise ValueError("HEAD_DIM must be multiple of 16")

    mod = _get_mod()

    return mod.int8_attention_forward(
        Q.contiguous(),
        K.contiguous(),
        V.contiguous(),
        timestep_scales.contiguous() if timestep_scales is not None else None,
        int(timestep),
        bool(causal),
    )

__all__ = ["int8_attention_forward"]