"""
Build script for INT8 fused attention CUDA kernel.

Usage:
    pip install -e .
or
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

cuda_sources = [
    os.path.join(THIS_DIR, "attention_int8.cu"),
]

cpp_sources = [
    os.path.join(THIS_DIR, "torch-ext", "torch_binding.cpp"),
]

# -----------------------------------------------------------------------------
# Multi-architecture support (A100, L40, H100)
# -----------------------------------------------------------------------------

extra_compile_args = {
    "cxx": [
        "-O3",
        "-std=c++17",
    ],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-lineinfo",

        # A100
        "-gencode=arch=compute_80,code=sm_80",

        # L40 (Ada Lovelace)
        "-gencode=arch=compute_89,code=sm_89",

        # H100
        "-gencode=arch=compute_90,code=sm_90",
    ],
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

setup(
    name="attention-int8-kernel",   # ⚠ do NOT use "kernels"
    version="0.1.0",
    description="Timestep-aware INT8 fused attention CUDA kernel for diffusion transformers",
    author="Your Name",
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
    ],
    ext_modules=[
        CUDAExtension(
            name="attention_int8._ops",
            sources=cpp_sources + cuda_sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)