from torch.utils.cpp_extension import load
import os
import torch

# Import the compiled extension
try:
    from . import _ops as ops
except ImportError:
    # Fallback for development/testing
    ops = None

if ops is not None:
    try:
        # Register as custom op (PyTorch 2.1+) for torch.compile compatibility
        @torch.library.custom_op("kernels::int8_attention_forward", mutates_args=())
        def _int8_attention_custom_op(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            timestep_scales: torch.Tensor | None = None,
            timestep: int = 0,
        ) -> torch.Tensor:
            return ops.int8_attention_forward(
                Q.contiguous(),
                K.contiguous(),
                V.contiguous(),
                timestep_scales.contiguous() if timestep_scales is not None else None,
                int(timestep),
            )

        @_int8_attention_custom_op.register_fake
        def _int8_attention_fake(
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            timestep_scales: torch.Tensor | None = None,
            timestep: int = 0,
        ) -> torch.Tensor:
            return Q.new_empty(Q.shape)

        _CUSTOM_OP_REGISTERED = True
    except (AttributeError, Exception):
        # torch.library.custom_op not available
        pass


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

def int8_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, timestep_scales: torch.Tensor=None, timestep: int=0) -> torch.Tensor:
    """Run the fused INT8 attention kernel.

    Args:
        Q,K,V: Tensors with shape [B,H,N,D], dtype=torch.float16, CUDA.
        timestep_scales: Optional float32 CUDA tensor lookup table.
        timestep: integer timestep index for lookup.

    Returns:
        output tensor of same shape and dtype as Q.
    """
    mod = _get_mod()
    if timestep_scales is None:
        return mod.int8_attention_forward(Q.contiguous(), K.contiguous(), V.contiguous(), None, int(timestep))
    else:
        return mod.int8_attention_forward(Q.contiguous(), K.contiguous(), V.contiguous(), timestep_scales.contiguous(), int(timestep))

__all__ = ["int8_attention_forward"]
