import torch
import attention_int8._ops as ops

print("="*60)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA not available"

device = torch.device("cuda")
torch.cuda.set_device(0)

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

# Minimal tensor (adjust dtype if your kernel expects different)
x = torch.randn(1, 64, device=device, dtype=torch.float16)

print("Input allocated:", x.shape)

# Replace `forward` with actual exported function name
# Example:
# y = ops.forward(x)
# print("Output:", y.shape)

print("Extension import successful.")
print("="*60)