import torch
from . import _attention_int8_cuda_dba582b_dirty
ops = torch.ops._attention_int8_cuda_dba582b_dirty

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_attention_int8_cuda_dba582b_dirty::{op_name}"
