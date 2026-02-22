#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Kernel launcher (defined in your .cu header)
// -----------------------------------------------------------------------------

cudaError_t launch_int8_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       O,
    const float*  timestep_scales,
    int  timestep,
    int  B, int H, int kv_H, int N, int D,
    bool causal,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// Input validation
// -----------------------------------------------------------------------------

static void check_qkv(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kHalf,
                name, " must be torch.float16");
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
}

static void check_ts(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "timestep_scales must be CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat,
                "timestep_scales must be float32");
    TORCH_CHECK(t.is_contiguous(),
                "timestep_scales must be contiguous");
}

// -----------------------------------------------------------------------------
// Forward
// -----------------------------------------------------------------------------

torch::Tensor int8_attention_forward(
    torch::Tensor Q,              // [B, H, N, D]
    torch::Tensor K,              // [B, kv_H, N, D]
    torch::Tensor V,              // [B, kv_H, N, D]
    c10::optional<torch::Tensor> timestep_scales_opt,
    int64_t timestep,
    bool causal)
{
    check_qkv(Q, "Q");
    check_qkv(K, "K");
    check_qkv(V, "V");

    TORCH_CHECK(Q.dim() == 4, "Q must be [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must be [B, kv_H, N, D]");
    TORCH_CHECK(V.dim() == 4, "V must be [B, kv_H, N, D]");

    const int B    = Q.size(0);
    const int H    = Q.size(1);
    const int N    = Q.size(2);
    const int D    = Q.size(3);

    const int B_k  = K.size(0);
    const int kv_H = K.size(1);
    const int N_k  = K.size(2);
    const int D_k  = K.size(3);

    TORCH_CHECK(B_k == B,  "K batch must match Q");
    TORCH_CHECK(N_k == N,  "K sequence length must match Q");
    TORCH_CHECK(D_k == D,  "K head dim must match Q");

    TORCH_CHECK(V.size(0) == B,     "V batch mismatch");
    TORCH_CHECK(V.size(1) == kv_H,  "V kv_H mismatch");
    TORCH_CHECK(V.size(2) == N,     "V sequence length mismatch");
    TORCH_CHECK(V.size(3) == D,     "V head dim mismatch");

    TORCH_CHECK(kv_H > 0 && kv_H <= H,
                "kv_H must satisfy 0 < kv_H <= H");

    TORCH_CHECK(D % 16 == 0,
                "HEAD_DIM must be multiple of 16 for WMMA");

    // Device guard (multi-GPU safe)
    at::cuda::CUDAGuard device_guard(Q.device());

    torch::Tensor O = torch::empty_like(Q);

    // -------------------------------------------------------------------------
    // Optional timestep scales
    // -------------------------------------------------------------------------

    const float* ts_ptr = nullptr;

    if (timestep_scales_opt.has_value()) {
        auto ts = timestep_scales_opt.value();
        check_ts(ts);
        ts_ptr = ts.data_ptr<float>();
    }

    // -------------------------------------------------------------------------
    // Raw pointers
    // -------------------------------------------------------------------------

    const __half* Q_ptr =
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());

    const __half* K_ptr =
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>());

    const __half* V_ptr =
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>());

    __half* O_ptr =
        reinterpret_cast<__half*>(O.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // -------------------------------------------------------------------------
    // Launch kernel
    // -------------------------------------------------------------------------

    cudaError_t err = launch_int8_attention(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        ts_ptr,
        static_cast<int>(timestep),
        B, H, kv_H, N, D,
        causal,
        stream);

    TORCH_CHECK(err == cudaSuccess,
                "INT8 attention kernel failed: ",
                cudaGetErrorString(err));

    return O;
}

// -----------------------------------------------------------------------------
// PyBind
// -----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward",
          &int8_attention_forward,
          "General INT8 diffusion attention (CUDA)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("timestep_scales") = c10::nullopt,
          py::arg("timestep") = 0,
          py::arg("causal") = false);
}