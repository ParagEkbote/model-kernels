{
  description = "INT8 Attention Kernel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs = { self, nixpkgs, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}