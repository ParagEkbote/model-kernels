  description = "Flake for kernel";

  inputs = {
    kernel-builder.url = "path:../../..";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}