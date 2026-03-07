from typing import Optional
import pathlib
import torch


# -----------------------------------------------------------------------------
# Load compiled extension
# -----------------------------------------------------------------------------

_pkg_dir = pathlib.Path(__file__).parent

_loaded = False
for file in _pkg_dir.iterdir():
    if file.name.startswith("_attention_int8") and file.suffix == ".so":
        torch.ops.load_library(str(file))
        _loaded = True
        break

if not _loaded:
    raise ImportError("INT8 attention extension library not found")


# -----------------------------------------------------------------------------
# Access dispatcher op
# -----------------------------------------------------------------------------

_ops = torch.ops.int8_attn


def int8_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    timestep_scales: Optional[torch.Tensor] = None,
    timestep: int = 0,
    causal: bool = False,
) -> torch.Tensor:
    return _ops.int8_attention_forward(Q, K, V, timestep_scales, timestep, causal)


__all__ = ["int8_attention_forward"]