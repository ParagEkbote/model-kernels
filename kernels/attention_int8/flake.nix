{
  description = "INT8 Attention Kernel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";  # stable channel
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs = { self, nixpkgs, kernel-builder }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    # Force Python 3.11 instead of default 3.13
    python = pkgs.python311;

  in
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;

      # override python used by kernel-builder
      python = python;
    };
}