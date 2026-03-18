.PHONY: all nix cachix submodules build

KERNEL_DIR := $(CURDIR)
CORES := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

NIX_FLAGS = --extra-experimental-features "nix-command flakes"

# Load nix environment
NIX_ENV = . $(HOME)/.nix-profile/etc/profile.d/nix.sh

all: nix cachix build


nix:
	@if command -v nix >/dev/null 2>&1; then \
		echo "Nix already installed"; \
	else \
		echo "Installing Nix..."; \
		curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install | sh -s -- --no-daemon; \
	fi

cachix:
	$(NIX_ENV) && \
	if ! command -v cachix >/dev/null 2>&1; then \
		echo "Installing Cachix..."; \
		nix $(NIX_FLAGS) profile add nixpkgs#cachix; \
	fi
	$(NIX_ENV) && cachix use huggingface


# -----------------------------------------------------------------------------
# Build kernel using local flake
# -----------------------------------------------------------------------------

build:
	@echo "Building kernel in $(KERNEL_DIR)"
	@echo "Detected CPU cores: $(CORES)"
	$(NIX_ENV) && nix $(NIX_FLAGS) run -L --max-jobs 1 --cores $(CORES) $(KERNEL_DIR)#build-and-copy

# -----------------------------------------------------------------------------
# Cleanup build artifacts
# -----------------------------------------------------------------------------

clean:
	rm -rf build/


# -----------------------------------------------------------------------------
# Environment info (useful for debugging CI)
# -----------------------------------------------------------------------------

info:
	@echo "Kernel directory: $(KERNEL_DIR)"
	@echo "Detected cores: $(CORES)"
	@command -v nix >/dev/null && nix --version || echo "Nix not installed"
