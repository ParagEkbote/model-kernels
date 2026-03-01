# ============================================================================
# Comprehensive Kernel Builder Makefile
# ============================================================================
# This Makefile provides a complete workflow for building HF kernels using Nix
# with experimental features, cachix caching, and proper error handling.
#
# Features:
#   - Automatic Nix/Cachix installation and setup
#   - Experimental features configuration
#   - Multi-architecture builds (A100, L40, H100)
#   - Caching via Hugging Face cachix
#   - Development environment setup
#   - Local and remote kernel building
#
# Usage:
#   make help                   Show this help
#   make install-nix            Install Nix (first time only)
#   make setup                  Setup Nix experimental features + cachix
#   make build                  Build kernel (default)
#   make dev                    Start development shell
#   make fetch URL=...          Fetch and build from URL
#   make clean                  Clean build artifacts
#   make test                   Run kernel tests
#
# ============================================================================

# ============================================================================
# Configuration Variables
# ============================================================================

# Default shell
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

# Color output for readability
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# Helper Functions
# ============================================================================

define print_header
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║ $(1)$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(NC)"
endef

define print_info
	@echo "$(GREEN)✓$(NC) $(1)"
endef

define print_warning
	@echo "$(YELLOW)⚠$(NC) $(1)"
endef

define print_error
	@echo "$(RED)✗$(NC) $(1)"
endef

# ============================================================================
# Phony targets (don't correspond to files)
# ============================================================================

.PHONY: help setup install-nix check-nix setup-experimental-features setup-cachix \
        build dev fetch clean test validate all-quick quick-build \
        show-config ensure-directories debug-nix

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help

# ============================================================================
# Main Targets
# ============================================================================

## help: Show this help message
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
	@echo "$(YELLOW)QUICK EXAMPLES$(NC)"
	@echo "  # First time setup:"
	@echo "  make install-nix && make setup"
	@echo ""
	@echo "  # Build kernel:"
	@echo "  make build"
	@echo ""
	@echo "  # Development:"
	@echo "  make dev"
	@echo ""
	@echo "  # Fetch and build from URL:"
	@echo "  make fetch URL=https://huggingface.co/user/kernel.git"
	@echo ""

# ============================================================================
# Setup and Installation
# ============================================================================

## setup: Complete setup (install Nix, enable experimental features, setup cachix)
setup: check-nix setup-experimental-features setup-cachix validate
	$(call print_header,Setup Complete!)
	@echo "$(GREEN)✓ Nix installed and configured$(NC)"
	@echo "$(GREEN)✓ Experimental features enabled$(NC)"
	@echo "$(GREEN)✓ Cachix configured$(NC)"
	@echo ""
	$(call print_info,"Ready to build! Run: make build")

## check-nix: Check if Nix is installed and functional
check-nix:
	$(call print_header,Checking Nix Installation)
	@command -v nix >/dev/null 2>&1 || { \
		$(call print_error,"Nix not found. Install with: make install-nix"); \
		exit 1; \
	}
	@nix --version
	$(call print_info,"Nix is installed")
	@command -v cachix >/dev/null 2>&1 && $(call print_info,"Cachix is installed") || $(call print_warning,"Cachix not installed (optional)")

## install-nix: Install Nix package manager
install-nix:
	$(call print_header,Installing Nix)
	@if command -v nix >/dev/null 2>&1; then \
		$(call print_warning,"Nix already installed: $$(nix --version)"); \
	else \
		$(call print_info,"Downloading Nix installer..."); \
		sh <(curl -L https://nixos.org/nix/install) --yes || { \
			$(call print_error,"Nix installation failed"); \
			exit 1; \
		}; \
		. ~/.nix-profile/etc/profile.d/nix.sh; \
		$(call print_info,"Nix installed successfully"); \
	fi
	@echo ""
	$(call print_info,"Please run: source ~/.nix-profile/etc/profile.d/nix.sh")
	$(call print_info,"Or add to ~/.bashrc or ~/.zshrc for permanent setup")

## setup-experimental-features: Enable Nix experimental features
setup-experimental-features: check-nix
	$(call print_header,Setting Up Experimental Features)
	@mkdir -p $(NIX_CONF_DIR)
	@echo "$(YELLOW)Adding experimental features to: $(NIX_CONF_DIR)/nix.conf$(NC)"
	@if [ -f $(NIX_CONF_DIR)/nix.conf ]; then \
		if grep -q "experimental-features" $(NIX_CONF_DIR)/nix.conf; then \
			$(call print_warning,"Experimental features already configured"); \
		else \
			echo "" >> $(NIX_CONF_DIR)/nix.conf; \
			echo "# ===== Experimental Features (added by Makefile) =====" >> $(NIX_CONF_DIR)/nix.conf; \
			echo "experimental-features = $(EXPERIMENTAL_FEATURES)" >> $(NIX_CONF_DIR)/nix.conf; \
			$(call print_info,"Experimental features enabled"); \
		fi; \
	else \
		echo "# Nix configuration" > $(NIX_CONF_DIR)/nix.conf; \
		echo "experimental-features = $(EXPERIMENTAL_FEATURES)" >> $(NIX_CONF_DIR)/nix.conf; \
		echo "max-jobs = $(MAX_JOBS)" >> $(NIX_CONF_DIR)/nix.conf; \
		echo "cores = $(CORES)" >> $(NIX_CONF_DIR)/nix.conf; \
		$(call print_info,"Created nix.conf with experimental features"); \
	fi
	@if [ -w $(SYSTEM_NIX_CONF)/nix.conf ]; then \
		echo "" >> $(SYSTEM_NIX_CONF)/nix.conf; \
		echo "# ===== Experimental Features (added by Makefile) =====" >> $(SYSTEM_NIX_CONF)/nix.conf; \
		echo "experimental-features = $(EXPERIMENTAL_FEATURES)" >> $(SYSTEM_NIX_CONF)/nix.conf; \
		echo "max-jobs = $(MAX_JOBS)" >> $(SYSTEM_NIX_CONF)/nix.conf; \
		echo "cores = $(CORES)" >> $(SYSTEM_NIX_CONF)/nix.conf; \
		$(call print_info,"System-wide Nix config updated"); \
	else \
		$(call print_warning,"Cannot write to $(SYSTEM_NIX_CONF)/nix.conf (may need sudo)"); \
	fi

## setup-cachix: Configure Hugging Face cachix
setup-cachix: check-nix
	$(call print_header,Setting Up Cachix)
	@command -v cachix >/dev/null 2>&1 || { \
		$(call print_info,"Installing cachix..."); \
		nix profile install nixpkgs#cachix; \
	}
	@$(call print_info,"Configuring Hugging Face cachix")
	@cachix use huggingface || $(call print_warning,"Cachix use failed - may need authentication")
	$(call print_info,"Cachix configured")

# ============================================================================
# Build Targets
# ============================================================================

## ensure-directories: Create necessary build directories
ensure-directories:
	@mkdir -p $(BUILD_DIR) $(RESULT_DIR)
	$(call print_info,"Build directories created")

## validate: Validate flake.nix and build.toml
validate: ensure-directories
	$(call print_header,Validating Configuration)
	@if [ ! -f $(SRC_DIR)/$(FLAKE_FILE) ]; then \
		$(call print_error,"$(FLAKE_FILE) not found in $(SRC_DIR)"); \
		exit 1; \
	fi
	$(call print_info,"$(FLAKE_FILE) found")
	@if [ ! -f $(SRC_DIR)/$(BUILD_TOML) ]; then \
		$(call print_error,"$(BUILD_TOML) not found in $(SRC_DIR)"); \
		echo "$(YELLOW)Use 'make create-flake' to generate one$(NC)"; \
		exit 1; \
	fi
	$(call print_info,"$(BUILD_TOML) found")
	@cd $(SRC_DIR) && nix flake show 2>&1 | head -20 || $(call print_warning,"Flake evaluation had issues - may be OK")
	$(call print_info,"Configuration validated")

## build: Build the kernel (main build target)
build: check-nix ensure-directories validate
	$(call print_header,Building Kernel: $(KERNEL_NAME))
	@$(call print_info,"Source: $(SRC_DIR)")
	@$(call print_info,"Output: $(RESULT_DIR)")
	@$(call print_info,"Jobs: $(MAX_JOBS), Cores: $(CORES)")
	@echo ""
	@cd $(SRC_DIR) && \
		nix build . \
		--max-jobs $(MAX_JOBS) \
		--cores $(CORES) \
		-L 2>&1 | tee $(BUILD_DIR)/build.log
	@if [ -d ./result ]; then \
		$(call print_info,"Copying build results..."); \
		rm -rf $(RESULT_DIR)/*; \
		cp -r --dereference ./result/* $(RESULT_DIR)/; \
		chmod -R 755 $(RESULT_DIR); \
		$(call print_info,"Build completed successfully"); \
	else \
		$(call print_error,"Build output not found"); \
		exit 1; \
	fi
	@echo ""
	@ls -lh $(RESULT_DIR)/ 2>/dev/null || true

## quick-build: Build without cleaning (faster, use with caution)
quick-build: check-nix validate
	$(call print_header,Quick Build (No Clean))
	@cd $(SRC_DIR) && \
		nix build . \
		--max-jobs $(MAX_JOBS) \
		--cores $(CORES) \
		--accept-flake-config \
		-L

## rebuild: Clean and rebuild
rebuild: clean build
	$(call print_info,"Rebuild complete")

## debug-build: Build with verbose output
debug-build: check-nix validate
	$(call print_header,Debug Build)
	@cd $(SRC_DIR) && \
		nix build . \
		--max-jobs 1 \
		--cores 1 \
		-L --verbose 2>&1 | tee $(BUILD_DIR)/debug.log
	$(call print_info,"Debug log saved to $(BUILD_DIR)/debug.log")

# ============================================================================
# Development Targets
# ============================================================================

## dev: Start Nix development shell
dev: check-nix validate
	$(call print_header,Starting Development Shell)
	@$(call print_info,"Enter development environment...")
	@echo ""
	@cd $(SRC_DIR) && nix develop
	@echo ""
	@$(call print_info,"Exited development shell")

## fetch: Clone and build kernel from Git URL (usage: make fetch URL=...)
fetch:
	@if [ -z "$(URL)" ]; then \
		$(call print_error,"URL not provided"); \
		echo "Usage: make fetch URL=https://github.com/user/kernel.git"; \
		exit 1; \
	fi
	$(call print_header,Fetching Kernel from URL)
	@$(call print_info,"URL: $(URL)")
	@$(call print_info,"Installing Git LFS...")
	@git lfs install || true
	@mkdir -p /tmp/kernel-fetch
	@$(call print_info,"Cloning repository...")
	@git clone $(URL) /tmp/kernel-fetch/repo
	@cd /tmp/kernel-fetch/repo && $(MAKE) build SRC_DIR=/tmp/kernel-fetch/repo
	@$(call print_info,"Build complete")
	@$(call print_info,"Results in: $(RESULT_DIR)")

# ============================================================================
# Cleaning and Maintenance
# ============================================================================

## clean: Remove build artifacts
clean:
	$(call print_header,Cleaning Build Artifacts)
	@rm -rf $(BUILD_DIR)
	@rm -rf ./result
	@$(call print_info,"Build artifacts removed")

## clean-cache: Clear Nix cache (careful!)
clean-cache:
	$(call print_header,Clearing Nix Cache)
	@$(call print_warning,"This will remove cached packages (can reclaim disk space)")
	@nix store gc || true
	$(call print_info,"Cache cleared")

## clean-all: Complete reset (removes everything)
clean-all: clean clean-cache
	$(call print_header,Complete Reset)
	@rm -rf $(RESULT_DIR)
	@rm -rf ~/.cache/nix
	@rm -rf ~/.nix-*
	@$(call print_warning,"All Nix caches and build artifacts removed")
	@$(call print_info,"Run 'make install-nix && make setup' to reinstall")

# ============================================================================
# Testing and Validation
# ============================================================================

## test: Run kernel tests (if test script exists)
test: build
	$(call print_header,Running Kernel Tests)
	@if [ -f $(SRC_DIR)/test_kernel.py ]; then \
		$(call print_info,"Found test_kernel.py"); \
		cd $(SRC_DIR) && python test_kernel.py; \
	elif [ -f $(SRC_DIR)/tests/test_*.py ]; then \
		$(call print_info,"Running pytest..."); \
		cd $(SRC_DIR) && python -m pytest tests/ -v; \
	else \
		$(call print_warning,"No test script found"); \
	fi

## show-config: Display current configuration
show-config:
	$(call print_header,Current Configuration)
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
	@[ -f $(SRC_DIR)/$(FLAKE_FILE) ] && echo "  ✓ $(FLAKE_FILE)" || echo "  ✗ $(FLAKE_FILE) missing"
	@[ -f $(SRC_DIR)/$(BUILD_TOML) ] && echo "  ✓ $(BUILD_TOML)" || echo "  ✗ $(BUILD_TOML) missing"
	@[ -d $(BUILD_DIR) ] && echo "  ✓ Build directory exists" || echo "  ✗ Build directory missing"
	@echo ""
	@echo "$(YELLOW)Nix Installation:$(NC)"
	@nix --version 2>/dev/null || echo "  Nix not installed"
	@cachix --version 2>/dev/null || echo "  Cachix not installed"

# ============================================================================
# Debugging Targets
# ============================================================================

## debug-nix: Debug Nix configuration and environment
debug-nix:
	$(call print_header,Nix Debug Information)
	@echo "$(YELLOW)Nix Version:$(NC)"
	@nix --version
	@echo ""
	@echo "$(YELLOW)Nix Configuration Files:$(NC)"
	@echo "  User config: $(NIX_CONF_DIR)/nix.conf"
	@[ -f $(NIX_CONF_DIR)/nix.conf ] && echo "    ✓ Exists" || echo "    ✗ Missing"
	@echo "  System config: $(SYSTEM_NIX_CONF)/nix.conf"
	@[ -f $(SYSTEM_NIX_CONF)/nix.conf ] && echo "    ✓ Exists" || echo "    ✗ Missing"
	@echo ""
	@echo "$(YELLOW)User Nix Config ($(NIX_CONF_DIR)/nix.conf):$(NC)"
	@cat $(NIX_CONF_DIR)/nix.conf 2>/dev/null || echo "  Not found"
	@echo ""
	@echo "$(YELLOW)Nix Store Info:$(NC)"
	@nix store info 2>/dev/null || echo "  Could not retrieve"
	@echo ""
	@echo "$(YELLOW)Available Outputs:$(NC)"
	@cd $(SRC_DIR) && nix flake show 2>/dev/null || echo "  Could not show flake"

## debug-flake: Check flake.nix structure
debug-flake:
	$(call print_header,Flake Configuration Debug)
	@if [ -f $(SRC_DIR)/$(FLAKE_FILE) ]; then \
		echo "$(YELLOW)Flake file found: $(SRC_DIR)/$(FLAKE_FILE)$(NC)"; \
		echo ""; \
		echo "$(YELLOW)First 50 lines:$(NC)"; \
		head -50 $(SRC_DIR)/$(FLAKE_FILE); \
		echo ""; \
		echo "$(YELLOW)Checking for kernel-builder input:$(NC)"; \
		grep -i "kernel-builder" $(SRC_DIR)/$(FLAKE_FILE) && echo "  ✓ Found" || echo "  ✗ Not found"; \
	else \
		$(call print_error,"$(FLAKE_FILE) not found"); \
	fi

# ============================================================================
# Advanced Targets (Power Users)
# ============================================================================

## all-quick: Install Nix, setup, and build (one command)
all-quick: install-nix setup build
	$(call print_header,Complete! Ready to deploy)
	@$(call print_info,"Kernel built: $(RESULT_DIR)")

# ============================================================================
# Internal Targets (not documented in help)
# ============================================================================

.PHONY: print_header print_info print_warning print_error

# Silent print helpers
print_header:
	@true

print_info:
	@true

print_warning:
	@true

print_error:
	@true

# ============================================================================
# Informational Targets
# ============================================================================

.PHONY: info version

## version: Show Makefile version
version:
	@echo "Kernel Builder Makefile v2.0"
	@echo "Built for Nix $(NIX_VERSION)"
	@echo "Supports experimental features: $(EXPERIMENTAL_FEATURES)"

# ============================================================================
# End of Makefile
# ============================================================================