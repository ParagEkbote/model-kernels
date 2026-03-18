# attention-int8

A custom kernel for PyTorch.

## Installation

```bash
pip install AINovice2005/attention-int8
```

## Usage

```python
import torch
from attention_int8 import attention_int8

# Create input tensor
x = torch.randn(1024, 1024, device="cuda")

# Run kernel
result = attention_int8(x)
```

## Development

### Building

```bash
nix develop
nix run .#build-and-copy
```

### Testing

```bash
nix develop .#test
pytest tests/
```

### Test as a `kernels` user

```bash
uv run example.py
```

## License

Apache 2.0
