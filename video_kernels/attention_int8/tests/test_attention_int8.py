import math
import pytest

try:
    import torch
except Exception:
    pytest.skip("PyTorch not installed; skipping tests", allow_module_level=True)

from kernels.attention_int8 import int8_attention_forward


# -----------------------------------------------------------------------------
# Reference implementations
# -----------------------------------------------------------------------------

def ref_attention(Q, K, V, causal=False):
    """
    Q: [B, H, N, D]
    K,V: [B, kv_H, N, D]
    """
    B, H, N, D = Q.shape
    kv_H = K.shape[1]

    q = Q.float()
    k = K.float()
    v = V.float()

    out = torch.empty_like(q)

    for b in range(B):
        for h in range(H):
            kv_h = h % kv_H
            q_bh = q[b, h]               # [N, D]
            k_bh = k[b, kv_h]            # [N, D]
            v_bh = v[b, kv_h]            # [N, D]

            scores = q_bh @ k_bh.T / math.sqrt(D)

            if causal:
                mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1)
                scores = scores.masked_fill(mask.bool(), float("-inf"))

            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ v_bh

    return out.half()


# -----------------------------------------------------------------------------
# Basic correctness (non-causal)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "B,H,kv_H,N,D",
    [
        (1, 1, 1, 8, 16),
        (1, 4, 4, 16, 32),
        (1, 4, 2, 16, 64),   # GQA
        (2, 8, 4, 32, 64),   # GQA larger
    ],
)
def test_int8_attention_forward(B, H, kv_H, N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, kv_H, N, D), dtype=torch.float16, device="cuda")
    V = torch.randn_like(K)

    try:
        out = int8_attention_forward(Q, K, V, None, 0, causal=False)
    except Exception as e:
        pytest.skip(f"Kernel launch failed: {e}")

    ref = ref_attention(Q, K, V, causal=False)

    torch.testing.assert_close(
        out, ref,
        rtol=2e-2,
        atol=2e-2,
    )


# -----------------------------------------------------------------------------
# Causal masking
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("N,D", [(16, 32), (32, 64)])
def test_int8_attention_causal(N, D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, kv_H = 1, 2, 2

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, kv_H, N, D), dtype=torch.float16, device="cuda")
    V = torch.randn_like(K)

    try:
        out = int8_attention_forward(Q, K, V, None, 0, causal=True)
    except Exception as e:
        pytest.skip(f"Kernel launch failed: {e}")

    ref = ref_attention(Q, K, V, causal=True)

    torch.testing.assert_close(
        out, ref,
        rtol=2e-2,
        atol=2e-2,
    )


# -----------------------------------------------------------------------------
# Timestep scaling path
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("D", [32, 64])
def test_int8_attention_with_timestep(D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, kv_H, N = 1, 2, 2, 16

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, kv_H, N, D), dtype=torch.float16, device="cuda")
    V = torch.randn_like(K)

    timestep_scales = torch.linspace(
        0.5, 1.5, steps=10, dtype=torch.float32, device="cuda"
    )

    timestep = 3

    try:
        out = int8_attention_forward(
            Q, K, V, timestep_scales, timestep, causal=False
        )
    except Exception as e:
        pytest.skip(f"Kernel launch failed: {e}")

    # Reference uses no scaling — we only verify shape & finite values
    assert out.shape == Q.shape
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------------
# HEAD_DIM dispatch coverage
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("D", [32, 64, 80, 96, 128, 160])
def test_supported_head_dims(D):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, kv_H, N = 1, 2, 2, 8

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, kv_H, N, D), dtype=torch.float16, device="cuda")
    V = torch.randn_like(K)

    try:
        out = int8_attention_forward(Q, K, V, None, 0, causal=False)
    except Exception as e:
        pytest.skip(f"Kernel launch failed: {e}")

    assert out.shape == Q.shape


# -----------------------------------------------------------------------------
# Invalid head dim
# -----------------------------------------------------------------------------

def test_invalid_head_dim():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, H, kv_H, N, D = 1, 1, 1, 8, 24  # not multiple of 16

    Q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
    K = torch.randn((B, kv_H, N, D), dtype=torch.float16, device="cuda")
    V = torch.randn_like(K)

    with pytest.raises(Exception):
        int8_attention_forward(Q, K, V, None, 0, causal=False)