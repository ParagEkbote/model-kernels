import torch
torch.ops.load_library("./torch-ext/attention_int8/_attention_int8_cuda_dba582b_dirty.abi3.so")

Q = torch.randn(1, 8, 2048, 64, dtype=torch.float16, device="cuda")
K = torch.randn(1, 8, 2048, 64, dtype=torch.float16, device="cuda")
V = torch.randn(1, 8, 2048, 64, dtype=torch.float16, device="cuda")

# Parag's kernel
O = torch.ops.int8_attn.int8_attention_forward(Q, K, V)

# PyTorch native attention (reference)
ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# Compare
diff_mean = (O.float() - ref.float()).abs().mean().item()
diff_max = (O.float() - ref.float()).abs().max().item()

print(f"Output shape: {O.shape}")
print(f"Mean difference: {diff_mean:.6f}")
print(f"Max difference: {diff_max:.6f}")

if diff_mean < 0.05:
    print("OK - results are correct")
else:
    print("PROBLEM - difference too large")