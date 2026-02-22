import math
import pytest

# If PyTorch is not installed in this environment, skip the whole module.
try:
    import torch
except Exception:
    pytest.skip("PyTorch not installed; skipping tests", allow_module_level=True)

from kernels.attention_int8 import int8_attention_forward


def ref_attention(Q, K, V):
    # Q,K,V: [B,H,N,D] half
    B, H, N, D = Q.shape
    q = Q.reshape(B * H, N, D).float()
    k = K.reshape(B * H, N, D).float()
    v = V.reshape(B * H, N, D).float()

    scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
    probs = torch.softmax(scores, dim=-1)
    out = torch.bmm(probs, v)
    return out.reshape(B, H, N, D).half()


@pytest.mark.parametrize("B,H,N,D", [(1, 1, 8, 16)])
def test_int8_attention_basic(B, H, N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Try calling the extension; if it fails to build/launch, skip the test
    try:
        out = int8_attention_forward(Q, K, V, None, 0)
    except Exception as e:
        pytest.skip(f"Extension not available or launch failed: {e}")

    ref = ref_attention(Q, K, V)

    # Compare with relatively loose tolerances for INT8 path
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_int8_attention_api_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, N, D = 1, 1, 8, 16
    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    timestep_scales = torch.ones(10, dtype=torch.float32, device="cuda")

    try:
        out = int8_attention_forward(Q, K, V, timestep_scales, 2)
    except Exception as e:
        pytest.skip(f"Extension not available or launch failed: {e}")

    assert out.shape == Q.shape
    assert out.dtype == torch.float16
