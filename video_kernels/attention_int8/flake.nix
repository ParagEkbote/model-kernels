# ============================================================================
# Production-Ready flake.nix for HF Kernel Builder
# ============================================================================
# This flake.nix is optimized for use with the provided Makefile.
# It supports:
#   - Experimental features (nix-command, flakes)
#   - Development shells with all dependencies
#   - Multi-architecture builds (A100, L40, H100)
#   - Cachix binary caching
#   - Both GitHub and local kernel-builder sources
#
# Usage:
#   nix flake show                       Show available outputs
#   nix develop                          Enter dev shell
#   nix build .                          Build the kernel
#   nix build . --override-input kernel-builder github:huggingface/kernels
#
# ============================================================================

{
  description = "HF Kernel - Production Build Configuration";

  inputs = {
    # Primary kernel-builder source (GitHub)
    kernel-builder.url = "github:huggingface/kernel-builder";
    
    # Uncomment for local development:
    # kernel-builder.url = "path:../kernel-builder";
    
    # Standard Nix inputs
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = 
    { self
    , kernel-builder
    , nixpkgs
    , flake-utils
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = kernel-builder.lib;
      in
      {
        # ====================================================================
        # Development Shell
        # ====================================================================
        # Usage: nix develop
        # Provides all tools needed for development
        devShells.default = pkgs.mkShell {
          name = "kernel-dev";
          
          buildInputs = with pkgs; [
            # Build system
            cmake
            ninja
            pkg-config
            
            # Compiler
            gcc
            gccStdenv.cc.cc.lib
            
            # CUDA
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cccl
            cudaPackages.cuda_cudart
            
            # Python + dependencies
            python311
            python311Packages.pip
            python311Packages.setuptools
            python311Packages.wheel
            
            # PyTorch
            python311Packages.torch-bin
            
            # Tools
            git
            git-lfs
          ];
          
          # Environment setup
          shellHook = ''
            echo "🚀 Kernel development environment loaded"
            echo "Available Python: $(python3 --version)"
            echo ""
            echo "Next steps:"
            echo "  1. Edit your kernel code"
            echo "  2. Run: python setup.py develop"
            echo "  3. Test: python test_kernel.py"
            echo "  4. Exit this shell: exit"
            echo ""
            
            # Setup CUDA paths
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
            export PATH="${pkgs.cudaPackages.cudatoolkit}/bin:$PATH"
            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH"
            export CPATH="${pkgs.cudaPackages.cudatoolkit}/include:$CPATH"
          '';
        };

        # ====================================================================
        # Build Outputs
        # ====================================================================
        # Usage: nix build . (or through Makefile: make build)
        packages = {
          default =
            if lib ? genKernelFlakeOutputs then
              # Use kernel-builder helpers if available
              (
                kernel-builder.lib.genKernelFlakeOutputs {
                  inherit self;
                  path = ./.;
                }
              ).packages.${system}.default
            else
              # Fallback: build with basic Python packaging
              pkgs.python311Packages.buildPythonPackage {
                pname = "kernel-int8-attention";
                version = "0.1.0";
                
                src = ./.;
                
                buildInputs = with pkgs; [
                  cudaPackages.cudatoolkit
                  cudaPackages.cuda_cccl
                  python311Packages.torch-bin
                ];
                
                propagatedBuildInputs = [
                  pkgs.python311Packages.torch-bin
                ];
                
                dontUseSetuptoolsBuild = false;
              };
        };

        # ====================================================================
        # Alternative Outputs (Optional, for advanced usage)
        # ====================================================================
        apps = {
          dev = {
            type = "app";
            program = "${pkgs.bash}/bin/bash";
            args = [ "-c" "nix develop" ];
          };
          
          build = {
            type = "app";
            program = "${pkgs.bash}/bin/bash";
            args = [ "-c" "nix build ." ];
          };
        };

        # ====================================================================
        # Flake Metadata
        # ====================================================================
      }
    )
    // {
      # System-agnostic metadata
      flakeMetadata = {
        name = "kernel-int8-attention";
        description = "INT8 Fused Attention CUDA Kernel for Diffusion Transformers";
        license = "Apache 2.0";
        homepage = "https://huggingface.co/kernels";
      };
    };

  # ====================================================================
  # Nix Configuration
  # ====================================================================
  nixConfig = {
    # Allow unfree packages (CUDA)
    allow-unfree = true;
    
    # Allow impure builds if needed
    allow-import-from-derivation = true;
    
    # Use HF cachix for faster builds
    substituters = [
      "https://huggingface.cachix.org"
      "https://cache.nixos.org"
    ];
    
    trusted-public-keys = [
      "huggingface.cachix.org-1:6jhC0V73P5F1N3BhJJoLaFGpLpRhLFHODXHXQPNHGmA="
      "cache.nixos.org-1:6NCHdD59X431o0gWypG7a9Tqo+COYvTW6WMH5quGcraft="
    ];
  };
}
