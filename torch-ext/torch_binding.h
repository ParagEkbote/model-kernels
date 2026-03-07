#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>

#include <vector>
#include <tuple>
#include <utility>
#include <string>

// ============================================================================
// INT8 Attention Binding - Header
// ============================================================================
// Complete PyTorch binding for INT8 fused attention kernel with:
// - Multi-backend support (CUDA, CPU fallback)
// - Comprehensive input validation
// - GQA/MQA support
// - Timestep-aware quantization
// - Causal masking
// ============================================================================

// ============================================================================
// Input Validation Functions
// ============================================================================

/**
 * @brief Validate tensor properties (dtype, device, contiguity)
 * @param tensor Tensor to validate
 * @param name Name of tensor for error messages
 * @param expected_dtype Expected tensor dtype (e.g., torch::kHalf for FP16)
 * @throws std::runtime_error if validation fails
 */
void validate_tensor_properties(
    const torch::Tensor& tensor,
    const std::string& name,
    torch::ScalarType expected_dtype);

/**
 * @brief Validate QKV tensor shapes and compatibility
 * @param Q Query tensor [B, H, N, D]
 * @param K Key tensor [B, kv_H, N, D]
 * @param V Value tensor [B, kv_H, N, D]
 * @throws std::runtime_error if shapes are incompatible
 */
void validate_qkv_shapes(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V);

/**
 * @brief Validate HEAD_DIM is supported by kernel
 * @param D Head dimension value
 * @throws std::runtime_error if D not in {32, 64, 80, 96, 128, 160, 256}
 */
void validate_head_dim(int64_t D);

/**
 * @brief Validate GQA/MQA constraint
 * @param H Number of query heads
 * @param kv_H Number of key/value heads
 * @throws std::runtime_error if constraint violated
 */
void validate_kva_constraint(int64_t H, int64_t kv_H);

/**
 * @brief Validate timestep_scales tensor if provided
 * @param timestep_scales Optional timestep scales tensor
 * @param timestep Current timestep index
 * @throws std::runtime_error if validation fails
 */
void validate_timestep_scales(
    const c10::optional<torch::Tensor>& timestep_scales,
    int64_t timestep);

// ============================================================================
// Main API
// ============================================================================

/**
 * @brief INT8 attention forward pass (main entry point)
 * 
 * Computes scaled dot-product attention with INT8 quantization:
 *   Attention(Q, K, V) = softmax(Q·K^T / sqrt(D) * scale) · V
 * 
 * Per-head INT8 quantization with optional timestep-aware scaling:
 *   inv_scale = 127 / (abs_max * timestep_scale)
 * 
 * Supports:
 * - Grouped Query Attention (kv_H < H): shares K/V heads across query groups
 * - Multi-Query Attention (kv_H = 1): single K/V head for all queries
 * - Causal masking: masks future positions (autoregressive generation)
 * - Online softmax: numerically stable, streaming computation
 * 
 * @param Q Query tensor [B, H, N, D] (float16)
 *          B = batch size
 *          H = number of query heads
 *          N = sequence length
 *          D = head dimension {32, 64, 80, 96, 128, 160, 256}
 * 
 * @param K Key tensor [B, kv_H, N, D] (float16)
 *          kv_H = number of key/value heads (kv_H <= H)
 * 
 * @param V Value tensor [B, kv_H, N, D] (float16)
 * 
 * @param timestep_scales Optional per-timestep scales [num_timesteps] (float32)
 *        If provided, scales[timestep] multiplies the quantization range
 *        for diffusion-aware quantization.
 *        If not provided, all timesteps use scale = 1.0
 * 
 * @param timestep Index into timestep_scales (ignored if scales not provided)
 *                 Must be 0 <= timestep < len(timestep_scales)
 * 
 * @param causal If true, apply causal masking: attend only to current and
 *               past positions (masks positions j > i where i is query index)
 *               Useful for autoregressive/language model generation.
 *               Default: false
 * 
 * @return Output tensor [B, H, N, D] (float16)
 * 
 * @throws std::runtime_error if:
 *   - Tensors not on same device (CUDA)
 *   - Tensors have mismatched dtypes
 *   - Tensors not contiguous
 *   - Shapes incompatible (B, H, N, D mismatch)
 *   - kv_H constraint violated (0 < kv_H <= H)
 *   - HEAD_DIM not in supported list
 *   - CUDA kernel launch fails
 * 
 * @example
 * ```cpp
 * torch::Tensor Q = torch::randn({2, 8, 1024, 64}, torch::kHalf).cuda();
 * torch::Tensor K = torch::randn({2, 2, 1024, 64}, torch::kHalf).cuda();
 * torch::Tensor V = torch::randn({2, 2, 1024, 64}, torch::kHalf).cuda();
 * 
 * // Standard attention
 * auto O = int8_attention_forward(Q, K, V);
 * 
 * // With timestep scaling (diffusion)
 * auto scales = torch::linspace(0.5, 2.0, 1000, torch::kFloat).cuda();
 * auto O_t = int8_attention_forward(Q, K, V, scales, 500);
 * 
 * // Causal (autoregressive)
 * auto O_causal = int8_attention_forward(Q, K, V, {}, 0, true);
 * ```
 */
torch::Tensor int8_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales = c10::nullopt,
    int64_t timestep = 0,
    bool causal = false);

/**
 * @brief INT8 attention backward pass (future extension)
 * 
 * Currently not implemented. For use when gradient computation needed.
 * 
 * @param grad_O Gradient of output [B, H, N, D]
 * @param Q Query tensor [B, H, N, D]
 * @param K Key tensor [B, kv_H, N, D]
 * @param V Value tensor [B, kv_H, N, D]
 * @param O Output tensor [B, H, N, D]
 * @param timestep_scales Optional timestep scales
 * @param timestep Current timestep
 * @param causal Causal masking flag
 * 
 * @return Tuple of (grad_Q, grad_K, grad_V)
 * 
 * @note Currently unimplemented. Returns error if called.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
int8_attention_backward(
    torch::Tensor grad_O,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    c10::optional<torch::Tensor> timestep_scales = c10::nullopt,
    int64_t timestep = 0,
    bool causal = false);

// ============================================================================
// Utility API
// ============================================================================

/**
 * @brief Get maximum shared memory requirement for given HEAD_DIM
 * 
 * Returns approximate shared memory in bytes needed for INT8 attention kernel
 * with given head dimension.
 * 
 * @param D Head dimension
 * @return Size in bytes; returns 0 if D not supported
 * 
 * @example
 * ```cpp
 * auto smem = get_int8_attention_smem_bytes(64);  // Returns ~47000
 * ```
 */
int64_t get_int8_attention_smem_bytes(int64_t D);

/**
 * @brief Check if HEAD_DIM is supported by kernel
 * 
 * @param D Head dimension
 * @return true if D in {32, 64, 80, 96, 128, 160, 256}
 */
bool is_head_dim_supported(int64_t D);

/**
 * @brief Get list of supported HEAD_DIM values
 * 
 * @return Vector of supported dimensions
 */
std::vector<int64_t> get_supported_head_dims();

std::pair<int64_t, int64_t> get_block_config(int64_t D);

int64_t get_occupancy_hint(int64_t D);

// ============================================================================
// Device Implementations
// ============================================================================

#ifdef TORCH_EXTENSION_CUDA

/**
 * @brief CUDA-specific INT8 attention implementation
 * @note Called only on CUDA devices via dispatcher
 */
torch::Tensor int8_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal);

#endif

#ifdef TORCH_EXTENSION_CPU

/**
 * @brief CPU fallback implementation (reference, slow)
 * @note Called only on CPU devices; for reference/validation only
 * @warning Very slow compared to CUDA; not recommended for production
 */
torch::Tensor int8_attention_cpu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal);

#endif