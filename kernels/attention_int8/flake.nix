# LTX-Video Custom Kernels - Nix Flake
#
# Usage:
#   # Update flake inputs (first time or when updating kernel-builder)
#   nix flake update
#
#   # Build all kernels
#   nix run .#build-and-copy --max-jobs 2 --cores 8 -L
#
#   # Enter development shell
#   nix develop
#
# For faster builds, add the HuggingFace cache:
#   cachix use huggingface
# Or without cachix:
#   nix run nixpkgs#cachix -- use huggingface

{
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      # Optional: Add Python test dependencies
      # pythonPackageOverlay = final: prev: {
      #   # Add test dependencies here
      # };
    };
}