"""
Build script for custom CUDA kernels.

Usage:
    pip install -e .

Or build only:
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA source files (relative paths)
cuda_sources = [
    "kernels/attention_int8.cu",
]

# C++ binding source (relative paths)
cpp_sources = [
    "torch-ext/torch_binding.cpp",
]

# Compiler flags optimized for H100 (sm_90)
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-arch=sm_90",  # H100 compute capability
        "-gencode=arch=compute_90,code=sm_90",
        "-lineinfo",  # Debug info
    ],
}

setup(
    name="kernels",
    version="0.1.0",
    description="Optimized CUDA kernels for LTX-Video on H100 GPUs",
    author="HuggingFace",
    packages=find_packages(where="torch-ext"),
    package_dir={"": "torch-ext"},
    ext_modules=[
        CUDAExtension(
            name="kernels._ops",
            sources=cpp_sources + cuda_sources,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
    extras_require={
        "diffusers": [
            "diffusers>=0.25.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
    },
)