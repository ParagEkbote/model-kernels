#include "torch_binding.h"
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <tuple>
#include <cmath>

// ============================================================================
// Kernel Launcher Declaration (from int8_attention_integrated.cu)
// ============================================================================

extern cudaError_t launch_int8_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       O,
    const float*  timestep_scales,
    int  timestep,
    int  B, int H, int kv_H, int N, int D,
    bool causal,
    cudaStream_t stream);

// ============================================================================
// Configuration Lookup Tables
// ============================================================================

constexpr int SUPPORTED_HEAD_DIMS[] = {32, 64, 80, 96, 128, 160, 256};
constexpr int NUM_SUPPORTED_HEAD_DIMS = 7;

// Block tile configurations per HEAD_DIM
struct BlockConfig {
    int D;
    int BQ;
    int BK;
    int smem_bytes;
    int min_blocks;
};

constexpr BlockConfig BLOCK_CONFIGS[] = {
    // D,   BQ,  BK,  smem_bytes, min_blocks
    {32,   64,  64,  28000,      2},
    {64,   64,  64,  47000,      2},
    {80,   64,  64,  57000,      2},
    {96,   64,  64,  66000,      2},
    {128,  64,  64,  81000,      1},
    {160,  32,  32,  56000,      1},
    {256,  32,  32,  83000,      1},
};

// ============================================================================
// Utility Functions: Configuration Lookup
// ============================================================================

bool is_head_dim_supported(int64_t D) {
    for (int i = 0; i < NUM_SUPPORTED_HEAD_DIMS; ++i) {
        if (SUPPORTED_HEAD_DIMS[i] == D) {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> get_supported_head_dims() {
    std::vector<int64_t> result;
    for (int i = 0; i < NUM_SUPPORTED_HEAD_DIMS; ++i) {
        result.push_back(SUPPORTED_HEAD_DIMS[i]);
    }
    return result;
}

std::pair<int64_t, int64_t> get_block_config(int64_t D) {
    for (const auto& cfg : BLOCK_CONFIGS) {
        if (cfg.D == D) {
            return {cfg.BQ, cfg.BK};
        }
    }
    return {0, 0};
}

int64_t get_occupancy_hint(int64_t D) {
    for (const auto& cfg : BLOCK_CONFIGS) {
        if (cfg.D == D) {
            return cfg.min_blocks;
        }
    }
    return 0;
}

int64_t get_int8_attention_smem_bytes(int64_t D) {
    for (const auto& cfg : BLOCK_CONFIGS) {
        if (cfg.D == D) {
            return cfg.smem_bytes;
        }
    }
    return 0;
}

// ============================================================================
// Validation Functions
// ============================================================================

void validate_tensor_properties(
    const torch::Tensor& tensor,
    const std::string& name,
    torch::ScalarType expected_dtype)
{
    TORCH_CHECK(tensor.is_cuda(),
                name, " must be a CUDA tensor");
    
    TORCH_CHECK(tensor.dtype() == expected_dtype,
                name, " must have dtype ", expected_dtype,
                " but got ", tensor.dtype());
    
    TORCH_CHECK(tensor.is_contiguous(),
                name, " must be contiguous");
}

void validate_qkv_shapes(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V)
{
    // Check dimensions
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-dimensional [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4-dimensional [B, kv_H, N, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4-dimensional [B, kv_H, N, D]");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D = Q.size(3);

    const int64_t B_k  = K.size(0);
    const int64_t kv_H = K.size(1);
    const int64_t N_k  = K.size(2);
    const int64_t D_k  = K.size(3);

    // Batch size
    TORCH_CHECK(B_k == B,
                "K batch dimension (", B_k, ") must match Q batch (", B, ")");

    // Sequence length
    TORCH_CHECK(N_k == N,
                "K sequence length (", N_k, ") must match Q sequence (", N, ")");

    // Head dimension
    TORCH_CHECK(D_k == D,
                "K head dimension (", D_k, ") must match Q head dimension (", D, ")");

    // V shape
    TORCH_CHECK(V.size(0) == B,
                "V batch dimension (", V.size(0), ") must match Q batch (", B, ")");
    TORCH_CHECK(V.size(1) == kv_H,
                "V kv_H dimension (", V.size(1), ") must match K kv_H (", kv_H, ")");
    TORCH_CHECK(V.size(2) == N,
                "V sequence length (", V.size(2), ") must match Q sequence (", N, ")");
    TORCH_CHECK(V.size(3) == D,
                "V head dimension (", V.size(3), ") must match Q head dimension (", D, ")");

    // Device consistency
    TORCH_CHECK(K.device() == Q.device(),
                "Q and K must be on the same device");
    TORCH_CHECK(V.device() == Q.device(),
                "Q and V must be on the same device");
}

void validate_head_dim(int64_t D) {
    TORCH_CHECK(D > 0, "HEAD_DIM must be positive");
    
    TORCH_CHECK(D % 16 == 0,
                "HEAD_DIM must be a multiple of 16 (for WMMA), got ", D);

    TORCH_CHECK(is_head_dim_supported(D),
                "HEAD_DIM=", D, " not supported. Supported values: ",
                "{32, 64, 80, 96, 128, 160, 256}");
}

void validate_kva_constraint(int64_t H, int64_t kv_H) {
    TORCH_CHECK(kv_H > 0,
                "kv_H must be positive, got ", kv_H);

    TORCH_CHECK(kv_H <= H,
                "kv_H (", kv_H, ") must be <= H (", H, ") for GQA/MQA");

    TORCH_CHECK(H % kv_H == 0,
                "H (", H, ") must be divisible by kv_H (", kv_H, ")");
}

void validate_timestep_scales(
    const c10::optional<torch::Tensor>& timestep_scales_opt,
    int64_t timestep)
{
    if (!timestep_scales_opt.has_value()) {
        // No scales provided, only validate timestep is non-negative
        TORCH_CHECK(timestep >= 0,
                    "timestep must be non-negative, got ", timestep);
        return;
    }

    auto ts = timestep_scales_opt.value();

    TORCH_CHECK(ts.is_cuda(),
                "timestep_scales must be a CUDA tensor");

    TORCH_CHECK(ts.dtype() == torch::kFloat,
                "timestep_scales must be float32, got ", ts.dtype());

    TORCH_CHECK(ts.is_contiguous(),
                "timestep_scales must be contiguous");

    TORCH_CHECK(ts.dim() == 1,
                "timestep_scales must be 1-dimensional, got shape ", ts.sizes());

    const int64_t T = ts.size(0);

    TORCH_CHECK(timestep >= 0 && timestep < T,
                "timestep (", timestep, ") must be in range [0, ", T, ")");

    // Validate scales are positive
    auto ts_cpu = ts.cpu();
    auto ts_data = ts_cpu.data_ptr<float>();
    for (int64_t i = 0; i < T; ++i) {
        TORCH_CHECK(ts_data[i] > 0.0f,
                    "timestep_scales[", i, "] must be positive, got ",
                    ts_data[i]);
    }
}

// ============================================================================
// CUDA Implementation
// ============================================================================

#ifdef TORCH_EXTENSION_CUDA

torch::Tensor int8_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal)
{
    // All validation done in int8_attention_forward before dispatch

    const int64_t B    = Q.size(0);
    const int64_t H    = Q.size(1);
    const int64_t N    = Q.size(2);
    const int64_t D    = Q.size(3);
    const int64_t kv_H = K.size(1);

    // Create output tensor
    torch::Tensor O = torch::empty_like(Q);

    // Get raw pointers (safe because tensors are validated/contiguous)
    const __half* Q_ptr = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
    const __half* K_ptr = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
    const __half* V_ptr = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
    __half* O_ptr = reinterpret_cast<__half*>(O.data_ptr<at::Half>());

    // Timestep scales pointer
    const float* ts_ptr = nullptr;
    if (timestep_scales.has_value()) {
        ts_ptr = timestep_scales.value().data_ptr<float>();
    }

    // Get current stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    cudaError_t err = launch_int8_attention(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        ts_ptr,
        static_cast<int>(timestep),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(kv_H),
        static_cast<int>(N),
        static_cast<int>(D),
        causal,
        stream);

    TORCH_CHECK(err == cudaSuccess,
                "INT8 attention kernel launch failed: ",
                cudaGetErrorString(err));

    return O;
}

#endif  // TORCH_EXTENSION_CUDA

// ============================================================================
// CPU Fallback Implementation (Reference Only)
// ============================================================================

#ifdef TORCH_EXTENSION_CPU

torch::Tensor int8_attention_cpu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal)
{
    // CPU implementation is a simple fallback for validation/testing
    // NOT optimized for production use

    TORCH_WARN_ONCE(
        "Running INT8 attention on CPU. This is a reference implementation "
        "and is VERY slow. Consider using a CUDA GPU for production.");

    const int64_t B    = Q.size(0);
    const int64_t H    = Q.size(1);
    const int64_t N    = Q.size(2);
    const int64_t D    = Q.size(3);
    const int64_t kv_H = K.size(1);

    // Convert to float32 for computation (easier on CPU)
    auto Q_f32 = Q.to(torch::kFloat32);
    auto K_f32 = K.to(torch::kFloat32);
    auto V_f32 = V.to(torch::kFloat32);

    // Compute Q·K^T / sqrt(D)
    auto scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // Reshape for matmul: [B*H, N, D] × [B*kv_H, D, N] → [B, H, N, N]
    auto Q_flat = Q_f32.reshape({B * H, N, D});
    auto K_flat = K_f32.reshape({B * kv_H, N, D});
    
    // Handle GQA by repeating kv heads if needed
    if (kv_H < H) {
        int repeat_factor = H / kv_H;
        std::vector<torch::Tensor> K_repeated;
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < repeat_factor; ++i) {
                K_repeated.push_back(K_flat[b * kv_H]);
            }
        }
        K_flat = torch::stack(K_repeated);
    }

    // QK = Q @ K^T with scaling
    auto QK = torch::matmul(Q_flat, K_flat.transpose(1, 2)) * scale;

    // Apply causal mask if needed
    if (causal) {
        auto mask = torch::triu(torch::ones_like(QK), 1) * -1e9;
        QK = QK + mask;
    }

    // Softmax
    auto attn_weights = torch::softmax(QK, -1);

    // Reshape V: [B*kv_H, N, D]
    auto V_flat = V_f32.reshape({B * kv_H, N, D});
    
    // Handle GQA for V
    if (kv_H < H) {
        int repeat_factor = H / kv_H;
        std::vector<torch::Tensor> V_repeated;
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t i = 0; i < repeat_factor; ++i) {
                V_repeated.push_back(V_flat[b * kv_H]);
            }
        }
        V_flat = torch::stack(V_repeated);
    }

    // Output = attn_weights @ V
    auto O_f32 = torch::matmul(attn_weights, V_flat);

    // Reshape back: [B*H, N, D] → [B, H, N, D]
    auto O = O_f32.reshape({B, H, N, D});

    // Convert back to float16
    return O.to(torch::kHalf);
}

#endif  // TORCH_EXTENSION_CPU

// ============================================================================
// Main API - Forward Pass with Dispatcher
// ============================================================================

torch::Tensor int8_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal)
{
    // ========================================================================
    // Input Validation Phase
    // ========================================================================

    // Check device (must be CUDA)
    TORCH_CHECK(Q.is_cuda(),
                "INT8 attention only supports CUDA tensors. "
                "Q is on device: ", Q.device());

    // Validate tensor properties
    validate_tensor_properties(Q, "Q", torch::kHalf);
    validate_tensor_properties(K, "K", torch::kHalf);
    validate_tensor_properties(V, "V", torch::kHalf);

    // Validate shapes
    validate_qkv_shapes(Q, K, V);

    // Extract dimensions
    const int64_t B    = Q.size(0);
    const int64_t H    = Q.size(1);
    const int64_t N    = Q.size(2);
    const int64_t D    = Q.size(3);
    const int64_t kv_H = K.size(1);

    // Validate HEAD_DIM
    validate_head_dim(D);

    // Validate GQA/MQA constraint
    validate_kva_constraint(H, kv_H);

    // Validate dimensions are reasonable
    TORCH_CHECK(B > 0 && B <= 256,
                "Batch size B must be in [1, 256], got ", B);

    TORCH_CHECK(H > 0 && H <= 1024,
                "Number of heads H must be in [1, 1024], got ", H);

    TORCH_CHECK(N > 0 && N <= 65536,
                "Sequence length N must be in [1, 65536], got ", N);

    // Validate timestep scales if provided
    validate_timestep_scales(timestep_scales, timestep);

    // ========================================================================
    // Device Guard & Dispatch
    // ========================================================================

    at::cuda::CUDAGuard device_guard(Q.device());

#ifdef TORCH_EXTENSION_CUDA
    return int8_attention_cuda(Q, K, V, timestep_scales, timestep, causal);
#else
    #ifdef TORCH_EXTENSION_CPU
    // Fallback to CPU if CUDA not available
    TORCH_WARN("CUDA backend not available, falling back to CPU (slow)");
    return int8_attention_cpu(Q, K, V, timestep_scales, timestep, causal);
    #else
    TORCH_CHECK(false,
                "INT8 attention requires either CUDA or CPU backend to be enabled");
    #endif
#endif
}

// ============================================================================
// Backward Pass (Unimplemented)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
int8_attention_backward(
    torch::Tensor grad_O,
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    c10::optional<torch::Tensor> timestep_scales,
    int64_t timestep,
    bool causal)
{
    TORCH_CHECK(false,
                "INT8 attention backward pass is not yet implemented. "
                "Use inference-only mode (torch.no_grad()) or implement backward.");

    // Dummy return to satisfy compiler
    return std::make_tuple(grad_O, grad_O, grad_O);
}

// ============================================================================
// PyTorch Library Registration (TORCH_LIBRARY_IMPL)
// ============================================================================

// Define the operation schema
TORCH_LIBRARY(int8_attn, m) {
    m.def("int8_attention_forward(Tensor Q, Tensor K, Tensor V, "
          "Tensor? timestep_scales=None, int timestep=0, bool causal=False) "
          "-> Tensor");
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(int8_attn, CUDA, m) {
    m.impl("int8_attention_forward", &int8_attention_forward);
}

// Register CPU implementation (if available)
#ifdef TORCH_EXTENSION_CPU
TORCH_LIBRARY_IMPL(int8_attn, CPU, m) {
    m.impl("int8_attention_forward", &int8_attention_forward);
}
#endif

// ============================================================================
// Backward Registration (if autograd needed in future)
// ============================================================================

namespace {

// Forward-only mode for now; backward can be added later
class Int8AttentionFunction : public torch::autograd::Function<Int8AttentionFunction> {
 public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor Q,
        torch::Tensor K,
        torch::Tensor V,
        c10::optional<torch::Tensor> timestep_scales,
        int64_t timestep,
        bool causal)
    {
        torch::NoGradGuard guard;
        return int8_attention_forward(Q, K, V, timestep_scales, timestep, causal);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs)
    {
        TORCH_CHECK(false,
                    "INT8 attention backward is not implemented. "
                    "Use with torch.no_grad()");
        return {};
    }
};

}  // anonymous namespace

// ============================================================================
// Python Binding (PyBind11 - if using pybind11)
// ============================================================================

// Note: Modern PyTorch uses TORCH_LIBRARY_IMPL above instead of pybind11
// But if you need pybind11 bindings, uncomment and adapt:
/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward",
          &int8_attention_forward,
          "INT8 fused attention forward pass",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("timestep_scales") = py::none(),
          py::arg("timestep") = 0,
          py::arg("causal") = false);

    m.def("is_head_dim_supported",
          &is_head_dim_supported,
          "Check if HEAD_DIM is supported");

    m.def("get_supported_head_dims",
          &get_supported_head_dims,
          "Get list of supported HEAD_DIM values");

    m.def("get_int8_attention_smem_bytes",
          &get_int8_attention_smem_bytes,
          "Get shared memory requirement for HEAD_DIM");
}
*/