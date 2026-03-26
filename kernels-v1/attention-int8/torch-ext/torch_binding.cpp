#include "torch_binding.h"
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <torch/library.h>
#include <tuple>
#include <vector>

// ============================================================================
// Kernel Launcher Declaration
// ============================================================================

extern cudaError_t launch_int8_attention(const __half *Q, const __half *K,
                                         const __half *V, __half *O,
                                         const float *timestep_scales,
                                         int timestep, int B, int H, int kv_H,
                                         int N, int D, bool causal,
                                         cudaStream_t stream);

// ============================================================================
// Supported HEAD_DIM values
// ============================================================================

constexpr int SUPPORTED_HEAD_DIMS[] = {32, 64, 80, 96, 128, 160, 256};
constexpr int NUM_SUPPORTED_HEAD_DIMS = 7;

// ============================================================================
// Utility Functions
// ============================================================================

bool is_head_dim_supported(int64_t D) {
  for (int i = 0; i < NUM_SUPPORTED_HEAD_DIMS; ++i)
    if (SUPPORTED_HEAD_DIMS[i] == D)
      return true;
  return false;
}

std::vector<int64_t> get_supported_head_dims() {
  return std::vector<int64_t>(SUPPORTED_HEAD_DIMS,
                              SUPPORTED_HEAD_DIMS + NUM_SUPPORTED_HEAD_DIMS);
}

// ============================================================================
// Validation
// ============================================================================

void validate_tensor(const torch::Tensor &t, const char *name,
                     torch::ScalarType dtype) {
  TORCH_CHECK(t.device().type() == torch::kCUDA, name,
              " must be on CUDA device, got ", t.device());

  TORCH_CHECK(t.dtype() == dtype, name, " must have dtype ", dtype, ", got ",
              t.dtype());

  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void validate_shapes(const torch::Tensor &Q, const torch::Tensor &K,
                     const torch::Tensor &V) {
  TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,D], got shape ", Q.sizes());
  TORCH_CHECK(K.dim() == 4, "K must be [B,kv_H,N,D], got shape ", K.sizes());
  TORCH_CHECK(V.dim() == 4, "V must be [B,kv_H,N,D], got shape ", V.sizes());

  TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
              "Batch size mismatch: Q=", Q.size(0), ", K=", K.size(0),
              ", V=", V.size(0));

  TORCH_CHECK(Q.size(2) == K.size(2) && Q.size(2) == V.size(2),
              "Sequence length mismatch: Q=", Q.size(2), ", K=", K.size(2),
              ", V=", V.size(2));

  TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3),
              "Head dimension mismatch: Q=", Q.size(3), ", K=", K.size(3),
              ", V=", V.size(3));
}

void validate_head_dim(int64_t D) {
  TORCH_CHECK(D % 16 == 0, "HEAD_DIM must be multiple of 16, got ", D);

    TORCH_CHECK(is_head_dim_supported(D),
        "Unsupported HEAD_DIM=", D,
        ". Supported: 32, 64, 80, 96, 128, 160, 256");
}

void validate_kv_constraint(int64_t H, int64_t kv_H) {
  TORCH_CHECK(kv_H > 0, "kv_H must be > 0, got ", kv_H);
  TORCH_CHECK(kv_H <= H, "kv_H must be <= H (kv_H=", kv_H, ", H=", H, ")");
  TORCH_CHECK(H % kv_H == 0, "H must be divisible by kv_H (H=", H,
              ", kv_H=", kv_H, ")");
}

void validate_timestep_scales(const c10::optional<torch::Tensor> &ts,
                              int64_t timestep, int64_t batch_size) {
  if (!ts.has_value())
    return;

  auto t = ts.value();

  TORCH_CHECK(t.device().type() == torch::kCUDA,
              "timestep_scales must be on CUDA device, got ", t.device());

  TORCH_CHECK(t.dtype() == torch::kFloat,
              "timestep_scales must be float32, got ", t.dtype());

  TORCH_CHECK(t.is_contiguous(), "timestep_scales must be contiguous");

  TORCH_CHECK(t.dim() == 1, "timestep_scales must be 1D, got shape ",
              t.sizes());

    TORCH_CHECK(t.size(0) == batch_size,
        "timestep_scales batch size mismatch: expected ", batch_size,
        ", got ", t.size(0));

  TORCH_CHECK(timestep >= 0, "timestep must be >= 0, got ", timestep);
}

// ============================================================================
// CUDA Implementation
// ============================================================================

torch::Tensor int8_attention_cuda(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V,
                                  c10::optional<torch::Tensor> timestep_scales,
                                  int64_t timestep, bool causal) {
  const int64_t B = Q.size(0);
  const int64_t H = Q.size(1);
  const int64_t N = Q.size(2);
  const int64_t D = Q.size(3);
  const int64_t kv_H = K.size(1);

  torch::Tensor O = torch::empty_like(Q);
  TORCH_CHECK(O.numel() > 0, "Failed to allocate output tensor");

  const __half *Q_ptr =
      reinterpret_cast<const __half *>(Q.data_ptr<at::Half>());
  const __half *K_ptr =
      reinterpret_cast<const __half *>(K.data_ptr<at::Half>());
  const __half *V_ptr =
      reinterpret_cast<const __half *>(V.data_ptr<at::Half>());
  __half *O_ptr = reinterpret_cast<__half *>(O.data_ptr<at::Half>());

  const float *ts_ptr = nullptr;
  if (timestep_scales.has_value()) {
    auto ts = timestep_scales.value();
    TORCH_CHECK(ts.device() == Q.device(),
                "timestep_scales must be on same device as Q");
    ts_ptr = ts.data_ptr<float>();
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cudaError_t err =
      launch_int8_attention(Q_ptr, K_ptr, V_ptr, O_ptr, ts_ptr, timestep, B, H,
                            kv_H, N, D, causal, stream);

  TORCH_CHECK(err == cudaSuccess,
              "INT8 attention kernel launch failed: ", cudaGetErrorString(err));

  return O;
}

// ============================================================================
// Main Forward (FP16 + BF16 support)
// ============================================================================

torch::Tensor
int8_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                       c10::optional<torch::Tensor> timestep_scales,
                       int64_t timestep, bool causal) {
  // Device check first
  TORCH_CHECK(Q.device().type() == torch::kCUDA,
              "Q must be on CUDA device, got ", Q.device());

  TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
              "Q, K, V must all be on same device");

  at::cuda::CUDAGuard device_guard(Q.device());

  auto original_dtype = Q.scalar_type();

  TORCH_CHECK(original_dtype == torch::kHalf ||
                  original_dtype == torch::kBFloat16,
              "Supported dtypes: float16 (fp16), bfloat16 (bf16). Got: ",
              original_dtype);

  bool convert_back_to_bf16 = false;

  if (original_dtype == torch::kBFloat16) {
    Q = Q.to(torch::kHalf);
    K = K.to(torch::kHalf);
    V = V.to(torch::kHalf);
    convert_back_to_bf16 = true;
  }

  validate_tensor(Q, "Q", torch::kHalf);
  validate_tensor(K, "K", torch::kHalf);
  validate_tensor(V, "V", torch::kHalf);

  validate_shapes(Q, K, V);

  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t kv_H = K.size(1);
  int64_t D = Q.size(3);

  validate_head_dim(D);
  validate_kv_constraint(H, kv_H);
  validate_timestep_scales(timestep_scales, timestep, B);

  torch::Tensor O =
      int8_attention_cuda(Q, K, V, timestep_scales, timestep, causal);

  if (convert_back_to_bf16)
    return O.to(torch::kBFloat16);

  return O;
}

TORCH_LIBRARY(int8_attn, m) {
  m.def("int8_attention_forward("
        "Tensor Q,"
        "Tensor K,"
        "Tensor V,"
        "Tensor? timestep_scales=None,"
        "int timestep=0,"
        "bool causal=False"
        ") -> Tensor"
    );
    m.impl("int8_attention_forward", torch::kCUDA, &int8_attention_forward);
}
