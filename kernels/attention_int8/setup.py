"""
Build script for INT8 fused attention CUDA kernel.

Supports multiple GPU architectures:
- NVIDIA A100 (compute_80)
- NVIDIA L40 Ada (compute_89)
- NVIDIA H100 (compute_90)
- NVIDIA A10 (compute_86)

Usage:
    pip install -e .
or
    python setup.py build_ext --inplace
or
    python -m pip install --upgrade pip setuptools wheel
    pip install -e .
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Suppress CUDA warnings
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Get absolute paths
THIS_DIR = Path(__file__).resolve().parent

# Define source files
CUDA_SOURCES = [
    THIS_DIR / "attention_int8.cu",
]

CPP_SOURCES = [
    THIS_DIR / "torch-ext" / "attention_int8" / "torch_binding.cpp",
]

# Convert to string paths
cuda_sources = [str(f) for f in CUDA_SOURCES if f.exists()]
cpp_sources = [str(f) for f in CPP_SOURCES if f.exists()]

# Validate that sources exist
missing_files = []
for f in CUDA_SOURCES + CPP_SOURCES:
    if not f.exists():
        missing_files.append(str(f))

if missing_files:
    print("⚠ WARNING: Missing source files:")
    for f in missing_files:
        print(f"  - {f}")
    print("\nMake sure you have:")
    print("  - attention_int8.cu (in root directory)")
    print("  - torch-ext/torch_binding.cpp")
    # Don't exit, just warn - user might be installing from wheel

# ============================================================================
# Compiler Flags Configuration
# ============================================================================

# C++ flags (common for all architectures)
cxx_flags = [
    "-O3",                    # Optimization level 3
    "-std=c++17",             # C++17 standard
    "-fPIC",                  # Position-independent code
    "-Wall",                  # Enable all warnings
    "-Wextra",
    "-Wpedantic",
    "-ffast-math",            # Fast math optimizations
]

# NVIDIA CUDA flags
# Reference: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
nvcc_flags = [
    "-O3",                    # Optimization level
    "-std=c++17",             # C++17 standard
    "--use_fast_math",        # Fast math functions
    "-U__CUDA_NO_HALF_OPERATORS__",    # Enable half precision
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-lineinfo",              # Keep line numbers for debugging
    "-Xcompiler",             # Pass to host compiler
    "-O3",
    "-Xcompiler",
    "-std=c++17",
]

# Architecture-specific code generation
# Format: -gencode arch=compute_XX,code=sm_XX
# arch = virtual architecture (PTX generation)
# code = real architecture (binary compilation)
architectures = [
    # Ampere (A100, A10)
    "-gencode=arch=compute_80,code=sm_80",
    
    # Ada (L40, RTX 5000)
    "-gencode=arch=compute_89,code=sm_89",
    
    # Hopper (H100, H200)
    "-gencode=arch=compute_90,code=sm_90",
    
    # (Optional) Older: Turing (A10, RTX 2080)
    # "-gencode=arch=compute_75,code=sm_75",
]

nvcc_flags.extend(architectures)

# Combine all NVCC flags
extra_compile_args = {
    "cxx": cxx_flags,
    "nvcc": nvcc_flags,
}

# ============================================================================
# Include Directories (optional, if using custom headers)
# ============================================================================

include_dirs = [
    str(THIS_DIR / "torch-ext"),  # Your custom headers
    # Torch headers are automatically included
    # CUDA headers are automatically included
]

# ============================================================================
# Setup Configuration
# ============================================================================

setup(
    # Package metadata
    name="attention-int8-kernel",
    version="0.1.0",
    description="INT8 fused attention CUDA kernel for diffusion transformers",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://huggingface.co/your-username/attention-int8-kernel",
    
    # Package discovery
    packages=find_packages(include=["attention_int8", "attention_int8.*"]),
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    
    # CUDA extension
    ext_modules=[
        CUDAExtension(
            name="attention_int8._ops",  # Import as: from attention_int8 import _ops
            sources=cpp_sources + cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            
            # Link with CUDA libraries
            
            # Whether to link statically or dynamically
            extra_link_args=[],
        )
    ],
    
    # Build options
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,           # Use Ninja build system (faster)
            no_python_abi_suffix=True,  # For better ABI compatibility
        )
    },
    
    # Don't zip the package (important for C++ extensions)
    zip_safe=False,
    
    # Include license and other files
    include_package_data=True,
)

# Print configuration info
print("=" * 70)
print("INT8 Kernel Build Configuration")
print("=" * 70)
print(f"CUDA Sources: {cuda_sources}")
print(f"C++ Sources: {cpp_sources}")
print(f"Include Dirs: {include_dirs}")
print(f"Architectures: {', '.join([a.split('=')[-1] for a in architectures])}")
print("=" * 70)