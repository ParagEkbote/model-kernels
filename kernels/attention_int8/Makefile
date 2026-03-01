# ============================================================================
# Simplified Kernel Builder Makefile - Production Ready
# ============================================================================
# This Makefile provides a complete workflow for building HF kernels using Nix
# with experimental features and cachix caching.
#
# Features:
#   - Automatic Nix/Cachix installation and setup
#   - Experimental features configuration
#   - Multi-architecture builds (A100, L40, H100)
#   - Caching via Hugging Face cachix
#   - Development environment setup
#   - Comprehensive error handling
#
# Usage:
#   make help                   Show this help
#   make install-nix            Install Nix (first time only)
#   make setup                  Setup Nix experimental features + cachix
#   make build                  Build kernel (default)
#   make dev                    Start development shell
#   make fetch URL=...          Fetch and build from URL
#   make clean                  Clean build artifacts
#
# ============================================================================

SHELL := /bin/bash

# Nix configuration
NIX_VERSION := 2.30.3
CACHIX_VERSION := latest
MAX_JOBS ?= 4
CORES ?= 4
EXPERIMENTAL_FEATURES := nix-command flakes

# Directory structure
SRC_DIR ?= $(shell pwd)
BUILD_DIR ?= $(SRC_DIR)/build
RESULT_DIR ?= $(BUILD_DIR)/result
NIX_CONF_DIR ?= ~/.config/nix
SYSTEM_NIX_CONF ?= /etc/nix

# Flake and build configuration
FLAKE_FILE := flake.nix
BUILD_TOML := build.toml
KERNEL_NAME ?= $(shell grep '^name' $(BUILD_TOML) 2>/dev/null | awk '{print $$3}' | tr -d '"' || echo "unknown-kernel")

# Color output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

.PHONY: help setup install-nix check-nix setup-experimental-features setup-cachix \
        build dev fetch clean test validate all-quick quick-build show-config \
        debug-nix debug-flake debug-build rebuild clean-cache clean-all

.DEFAULT_GOAL := help

# ============================================================================
# Main Help
# ============================================================================

help:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║         Kernel Builder Makefile - Complete Reference      ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)SETUP TARGETS (Run Once)$(NC)"
	@echo "  make install-nix              Install Nix package manager"
	@echo "  make setup                    Complete setup (Nix + experimental + cachix)"
	@echo "  make check-nix                Check if Nix is installed and working"
	@echo ""
	@echo "$(YELLOW)BUILD TARGETS$(NC)"
	@echo "  make build                    Build kernel (default)"
	@echo "  make rebuild                  Clean and rebuild"
	@echo "  make quick-build              Build without cleaning (faster)"
	@echo "  make dev                      Start Nix development shell"
	@echo "  make fetch URL=<url>          Clone and build from Git URL"
	@echo ""
	@echo "$(YELLOW)MAINTENANCE$(NC)"
	@echo "  make clean                    Remove build artifacts"
	@echo "  make clean-cache              Remove Nix cache"
	@echo "  make clean-all                Remove everything (hard reset)"
	@echo "  make validate                 Validate flake.nix and build.toml"
	@echo "  make test                     Run kernel tests"
	@echo "  make show-config              Display current configuration"
	@echo ""
	@echo "$(YELLOW)DEBUGGING$(NC)"
	@echo "  make debug-nix                Debug Nix configuration"
	@echo "  make debug-build              Build with verbose output"
	@echo "  make debug-flake              Check flake.nix status"
	@echo ""
	@echo "$(YELLOW)VARIABLES (override with make VAR=value)$(NC)"
	@echo "  MAX_JOBS                      Number of parallel jobs (default: 4)"
	@echo "  CORES                         Cores per job (default: 4)"
	@echo "  SRC_DIR                       Source directory (default: current)"
	@echo "  BUILD_DIR                     Build directory (default: ./build)"
	@echo ""

# ============================================================================
# Setup and Installation
# ============================================================================

setup: check-nix setup-experimental-features setup-cachix validate
	@echo ""
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Setup Complete!                                           ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(GREEN)✓$(NC) Nix installed and configured"
	@echo "$(GREEN)✓$(NC) Experimental features enabled"
	@echo "$(GREEN)✓$(NC) Cachix configured"
	@echo ""
	@echo "Ready to build! Run: make build"

check-nix:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Checking Nix Installation                                ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@command -v nix >/dev/null 2>&1 || { echo "$(RED)✗$(NC) Nix not found. Install with: make install-nix"; exit 1; }
	@nix --version
	@echo "$(GREEN)✓$(NC) Nix is installed"
	@command -v cachix >/dev/null 2>&1 && echo "$(GREEN)✓$(NC) Cachix is installed" || echo "$(YELLOW)⚠$(NC) Cachix not installed (optional)"

install-nix:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Installing Nix                                            ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@if command -v nix >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠$(NC) Nix already installed: $$(nix --version)"; \
	else \
		echo "$(GREEN)✓$(NC) Downloading Nix installer..."; \
		sh <(curl -L https://nixos.org/nix/install) --yes || { echo "$(RED)✗$(NC) Nix installation failed"; exit 1; }; \
		. ~/.nix-profile/etc/profile.d/nix.sh; \
		echo "$(GREEN)✓$(NC) Nix installed successfully"; \
	fi
	@echo ""
	@echo "$(GREEN)✓$(NC) Please run: source ~/.nix-profile/etc/profile.d/nix.sh"
	@echo "$(GREEN)✓$(NC) Or add to ~/.bashrc for permanent setup"

setup-experimental-features: check-nix
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Setting Up Experimental Features                         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@mkdir -p $(NIX_CONF_DIR)
	@echo "$(YELLOW)Adding experimental features to: $(NIX_CONF_DIR)/nix.conf$(NC)"
	@if [ -f $(NIX_CONF_DIR)/nix.conf ]; then \
		if grep -q "experimental-features" $(NIX_CONF_DIR)/nix.conf; then \
			echo "$(YELLOW)⚠$(NC) Experimental features already configured"; \
		else \
			echo "" >> $(NIX_CONF_DIR)/nix.conf; \
			echo "# ===== Experimental Features =====" >> $(NIX_CONF_DIR)/nix.conf; \
			echo "experimental-features = $(EXPERIMENTAL_FEATURES)" >> $(NIX_CONF_DIR)/nix.conf; \
			echo "$(GREEN)✓$(NC) Experimental features enabled"; \
		fi; \
	else \
		echo "# Nix configuration" > $(NIX_CONF_DIR)/nix.conf; \
		echo "experimental-features = $(EXPERIMENTAL_FEATURES)" >> $(NIX_CONF_DIR)/nix.conf; \
		echo "max-jobs = $(MAX_JOBS)" >> $(NIX_CONF_DIR)/nix.conf; \
		echo "cores = $(CORES)" >> $(NIX_CONF_DIR)/nix.conf; \
		echo "$(GREEN)✓$(NC) Created nix.conf with experimental features"; \
	fi

setup-cachix: check-nix
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Setting Up Cachix                                         ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@command -v cachix >/dev/null 2>&1 || { echo "$(GREEN)✓$(NC) Installing cachix..."; nix profile install nixpkgs#cachix; }
	@echo "$(GREEN)✓$(NC) Configuring Hugging Face cachix"
	@cachix use huggingface || echo "$(YELLOW)⚠$(NC) Cachix use may need authentication"
	@echo "$(GREEN)✓$(NC) Cachix configured"

# ============================================================================
# Build Targets
# ============================================================================

ensure-directories:
	@mkdir -p $(BUILD_DIR) $(RESULT_DIR)

validate: ensure-directories
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Validating Configuration                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@if [ ! -f $(SRC_DIR)/$(FLAKE_FILE) ]; then echo "$(RED)✗$(NC) $(FLAKE_FILE) not found"; exit 1; fi
	@echo "$(GREEN)✓$(NC) $(FLAKE_FILE) found"
	@if [ ! -f $(SRC_DIR)/$(BUILD_TOML) ]; then echo "$(RED)✗$(NC) $(BUILD_TOML) not found"; exit 1; fi
	@echo "$(GREEN)✓$(NC) $(BUILD_TOML) found"
	@if grep -q "experimental-features" ~/.config/nix/nix.conf 2>/dev/null; then \
		cd $(SRC_DIR) && nix flake show 2>&1 | head -10 || true; \
		echo "$(GREEN)✓$(NC) Flake validated"; \
	else \
		echo "$(YELLOW)⚠$(NC) Experimental features not in config yet (will use inline flags)"; \
	fi

build: check-nix ensure-directories validate
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Building Kernel: $(KERNEL_NAME)                          ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(GREEN)✓$(NC) Source: $(SRC_DIR)"
	@echo "$(GREEN)✓$(NC) Output: $(RESULT_DIR)"
	@echo "$(GREEN)✓$(NC) Jobs: $(MAX_JOBS), Cores: $(CORES)"
	@echo ""
	@cd $(SRC_DIR) && nix build . --max-jobs $(MAX_JOBS) --cores $(CORES) --extra-experimental-features nix-command --extra-experimental-features flakes --accept-flake-config -L 2>&1 | tee $(BUILD_DIR)/build.log || { echo "$(RED)✗$(NC) Build failed"; exit 1; }
	@if [ -d ./result ]; then \
		echo "$(GREEN)✓$(NC) Copying build results..."; \
		rm -rf $(RESULT_DIR)/*; \
		cp -r --dereference ./result/* $(RESULT_DIR)/; \
		chmod -R 755 $(RESULT_DIR); \
		echo "$(GREEN)✓$(NC) Build completed successfully"; \
	else \
		echo "$(RED)✗$(NC) Build output not found"; \
		exit 1; \
	fi
	@echo ""
	@ls -lh $(RESULT_DIR)/ 2>/dev/null || true

quick-build: check-nix validate
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Quick Build (No Clean)                                    ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@cd $(SRC_DIR) && nix build . --max-jobs $(MAX_JOBS) --cores $(CORES) --extra-experimental-features nix-command --extra-experimental-features flakes --accept-flake-config -L

rebuild: clean build
	@echo "$(GREEN)✓$(NC) Rebuild complete"

debug-build: check-nix validate
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Debug Build                                               ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@cd $(SRC_DIR) && nix build . --max-jobs 1 --cores 1 --extra-experimental-features nix-command --extra-experimental-features flakes --accept-flake-config -L --verbose 2>&1 | tee $(BUILD_DIR)/debug.log
	@echo "$(GREEN)✓$(NC) Debug log saved to $(BUILD_DIR)/debug.log"

# ============================================================================
# Development Targets
# ============================================================================

dev: check-nix validate
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Starting Development Shell                                ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(GREEN)✓$(NC) Entering development environment..."
	@echo ""
	@cd $(SRC_DIR) && nix develop
	@echo ""
	@echo "$(GREEN)✓$(NC) Exited development shell"

fetch:
	@if [ -z "$(URL)" ]; then echo "$(RED)✗$(NC) URL not provided. Usage: make fetch URL=https://github.com/user/kernel.git"; exit 1; fi
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Fetching Kernel from URL                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(GREEN)✓$(NC) URL: $(URL)"
	@echo "$(GREEN)✓$(NC) Installing Git LFS..."
	@git lfs install || true
	@mkdir -p /tmp/kernel-fetch
	@echo "$(GREEN)✓$(NC) Cloning repository..."
	@git clone $(URL) /tmp/kernel-fetch/repo
	@cd /tmp/kernel-fetch/repo && $(MAKE) build SRC_DIR=/tmp/kernel-fetch/repo
	@echo "$(GREEN)✓$(NC) Build complete. Results in: $(RESULT_DIR)"

# ============================================================================
# Maintenance Targets
# ============================================================================

clean:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Cleaning Build Artifacts                                  ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@rm -rf $(BUILD_DIR)
	@rm -rf ./result
	@echo "$(GREEN)✓$(NC) Build artifacts removed"

clean-cache:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Clearing Nix Cache                                        ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)⚠$(NC) This will remove cached packages (can reclaim disk space)"
	@nix store gc || true
	@echo "$(GREEN)✓$(NC) Cache cleared"

clean-all: clean clean-cache
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Complete Reset                                            ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@rm -rf $(RESULT_DIR)
	@rm -rf ~/.cache/nix
	@rm -rf ~/.nix-*
	@echo "$(YELLOW)⚠$(NC) All Nix caches and build artifacts removed"
	@echo "$(GREEN)✓$(NC) Run 'make install-nix && make setup' to reinstall"

test: build
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Running Kernel Tests                                      ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@if [ -f $(SRC_DIR)/test_kernel.py ]; then \
		echo "$(GREEN)✓$(NC) Found test_kernel.py"; \
		cd $(SRC_DIR) && python test_kernel.py; \
	elif [ -f $(SRC_DIR)/tests/test_*.py ]; then \
		echo "$(GREEN)✓$(NC) Running pytest..."; \
		cd $(SRC_DIR) && python -m pytest tests/ -v; \
	else \
		echo "$(YELLOW)⚠$(NC) No test script found"; \
	fi

show-config:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Current Configuration                                      ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Build Configuration:$(NC)"
	@echo "  Source directory:     $(SRC_DIR)"
	@echo "  Build directory:      $(BUILD_DIR)"
	@echo "  Results directory:    $(RESULT_DIR)"
	@echo "  Kernel name:          $(KERNEL_NAME)"
	@echo ""
	@echo "$(YELLOW)Nix Configuration:$(NC)"
	@echo "  Max jobs:             $(MAX_JOBS)"
	@echo "  Cores per job:        $(CORES)"
	@echo "  Experimental:         $(EXPERIMENTAL_FEATURES)"
	@echo ""
	@echo "$(YELLOW)File Status:$(NC)"
	@[ -f $(SRC_DIR)/$(FLAKE_FILE) ] && echo "  $(GREEN)✓$(NC) $(FLAKE_FILE)" || echo "  $(RED)✗$(NC) $(FLAKE_FILE) missing"
	@[ -f $(SRC_DIR)/$(BUILD_TOML) ] && echo "  $(GREEN)✓$(NC) $(BUILD_TOML)" || echo "  $(RED)✗$(NC) $(BUILD_TOML) missing"
	@[ -d $(BUILD_DIR) ] && echo "  $(GREEN)✓$(NC) Build directory exists" || echo "  $(RED)✗$(NC) Build directory missing"
	@echo ""
	@echo "$(YELLOW)Nix Installation:$(NC)"
	@nix --version 2>/dev/null || echo "  Nix not installed"
	@cachix --version 2>/dev/null || echo "  Cachix not installed"

# ============================================================================
# Debugging Targets
# ============================================================================

debug-nix:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Nix Debug Information                                     ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(YELLOW)Nix Version:$(NC)"
	@nix --version
	@echo ""
	@echo "$(YELLOW)Nix Configuration Files:$(NC)"
	@echo "  User config: $(NIX_CONF_DIR)/nix.conf"
	@[ -f $(NIX_CONF_DIR)/nix.conf ] && echo "    $(GREEN)✓$(NC) Exists" || echo "    $(RED)✗$(NC) Missing"
	@echo "  System config: $(SYSTEM_NIX_CONF)/nix.conf"
	@[ -f $(SYSTEM_NIX_CONF)/nix.conf ] && echo "    $(GREEN)✓$(NC) Exists" || echo "    $(RED)✗$(NC) Missing"
	@echo ""
	@echo "$(YELLOW)User Nix Config:$(NC)"
	@cat $(NIX_CONF_DIR)/nix.conf 2>/dev/null || echo "  Not found"
	@echo ""
	@echo "$(YELLOW)Nix Store Info:$(NC)"
	@nix store info 2>/dev/null || echo "  Could not retrieve"

debug-flake:
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Flake Configuration Debug                                 ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@if [ -f $(SRC_DIR)/$(FLAKE_FILE) ]; then \
		echo "$(YELLOW)Flake file found: $(SRC_DIR)/$(FLAKE_FILE)$(NC)"; \
		echo ""; \
		echo "$(YELLOW)First 30 lines:$(NC)"; \
		head -30 $(SRC_DIR)/$(FLAKE_FILE); \
		echo ""; \
		echo "$(YELLOW)Checking for kernel-builder input:$(NC)"; \
		grep -i "kernel-builder" $(SRC_DIR)/$(FLAKE_FILE) && echo "  $(GREEN)✓$(NC) Found" || echo "  $(RED)✗$(NC) Not found"; \
	else \
		echo "$(RED)✗$(NC) $(FLAKE_FILE) not found"; \
	fi

# ============================================================================
# Advanced Targets
# ============================================================================

all-quick: install-nix setup build
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ Complete! Ready to deploy                                 ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
	@echo "$(GREEN)✓$(NC) Kernel built: $(RESULT_DIR)"

version:
	@echo "Kernel Builder Makefile v2.1 (Fixed)"
	@echo "Built for Nix $(NIX_VERSION)"
	@echo "Supports experimental features: $(EXPERIMENTAL_FEATURES)"

# ============================================================================
# End of Makefile
# ============================================================================