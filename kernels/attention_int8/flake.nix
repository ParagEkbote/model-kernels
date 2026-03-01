{
  description = "HF Kernel - INT8 Attention Build Configuration";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pythonPkgs = pkgs.python311Packages;

        # 🔧 Override torch-bin to disable strict runtime dependency checks
        torchFixed =
          pythonPkgs.torch-bin.overridePythonAttrs (old: {
            dontCheckRuntimeDeps = true;
          });

      in
      {
        # ==========================
        # Development Shell
        # ==========================
        devShells.default = pkgs.mkShell {
          name = "kernel-dev";

          buildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            gcc
            git
            cudaPackages.cudatoolkit
            python311
            pythonPkgs.pip
            pythonPkgs.setuptools
            torchFixed
          ];

          shellHook = ''
            export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH"
            echo "Environment ready. Run 'python setup.py develop'."
          '';
        };

        # ==========================
        # Production Build
        # ==========================
        packages.default = pythonPkgs.buildPythonPackage {
          pname = "kernel-int8-attention";
          version = "0.1.0";
          format = "setuptools";

          src = ./.;

          nativeBuildInputs = with pkgs; [
            cudaPackages.cuda_nvcc
            which
            ninja
            pythonPkgs.setuptools
            pythonPkgs.wheel
          ];

          buildInputs = with pkgs; [
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cccl
            cudaPackages.cuda_cudart
          ];

          propagatedBuildInputs = [
            torchFixed
          ];

          CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";

          doCheck = false;
          dontUseCmakeConfigure = true;
        };
      }
    ) // {
      nixConfig = {
        allow-unfree = true;
        substituters = [
          "https://huggingface.cachix.org"
          "https://cache.nixos.org"
        ];
        trusted-public-keys = [
          "huggingface.cachix.org-1:6jhC0V73P5F1N3BhJJoLaFGpLpRhLFHODXHXQPNHGmA="
          "cache.nixos.org-1:6NCHdD59X431o0gWypG7a9Tqo+COYvTW6WMH5quGzA="
        ];
      };
    };
}