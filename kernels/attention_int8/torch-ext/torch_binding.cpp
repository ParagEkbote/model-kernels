#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Kernel declaration (matches rewritten kernel)
// ----------------------------------------------------------------------------

extern "C" __global__ void int8_attention_general_kernel(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    const float* timestep_scales,
    int timestep,
    int B, int H, int N, int D
);

// ----------------------------------------------------------------------------
// Constants (must match kernel)
// ----------------------------------------------------------------------------

constexpr int BLOCK_Q = 64;
constexpr int BLOCK_K = 64;
constexpr int THREADS = 256;
constexpr int MAX_D   = 256;

// ----------------------------------------------------------------------------
// Input Validation
// ----------------------------------------------------------------------------

static void check_input(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kHalf,
                name, " must be torch.float16");
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

torch::Tensor int8_attention_forward(
    torch::Tensor Q,  // [B, H, N, D]
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> timestep_scales_opt,
    int64_t timestep
) {
    check_input(Q, "Q");
    check_input(K, "K");
    check_input(V, "V");

    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    TORCH_CHECK(D <= MAX_D,
                "Head dimension D exceeds MAX_D (", MAX_D, ")");

    // Device guard (multi-GPU safe)
    at::cuda::CUDAGuard device_guard(Q.device());

    torch::Tensor O = torch::empty_like(Q);

    // ----------------------------------------------------------------------------
    // Optional timestep scales
    // ----------------------------------------------------------------------------

    const float* timestep_scales_ptr = nullptr;

    if (timestep_scales_opt.has_value()) {
        auto ts = timestep_scales_opt.value();
        TORCH_CHECK(ts.is_cuda(),
                    "timestep_scales must be CUDA tensor");
        TORCH_CHECK(ts.dtype() == torch::kFloat,
                    "timestep_scales must be float32");
        TORCH_CHECK(ts.is_contiguous(),
                    "timestep_scales must be contiguous");

        timestep_scales_ptr = ts.data_ptr<float>();
    }

    // ----------------------------------------------------------------------------
    // Launch Configuration
    // ----------------------------------------------------------------------------

    dim3 block(THREADS);

    // grid.z tiles over Q dimension
    int q_tiles = (N + BLOCK_Q - 1) / BLOCK_Q;

    dim3 grid(B, H, q_tiles);

    // ----------------------------------------------------------------------------
    // Shared Memory Calculation
    // ----------------------------------------------------------------------------
    // Q_i8      : BLOCK_Q * MAX_D
    // K_i8      : BLOCK_K * MAX_D
    // V_tile    : BLOCK_K * MAX_D (__half)
    // QK_i32    : BLOCK_Q * BLOCK_K (int32)
    // No BLOCK_Q * N allocation anymore

    size_t shmem = 0;
    shmem += BLOCK_Q * MAX_D * sizeof(int8_t);
    shmem += BLOCK_K * MAX_D * sizeof(int8_t);
    shmem += BLOCK_K * MAX_D * sizeof(__half);
    shmem += BLOCK_Q * BLOCK_K * sizeof(int32_t);

    // ----------------------------------------------------------------------------
    // Raw pointers
    // ----------------------------------------------------------------------------

    const __half* Q_ptr =
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());

    const __half* K_ptr =
        reinterpret_cast<const __half*>(K.data_ptr<at::Half>());

    const __half* V_ptr =
        reinterpret_cast<const __half*>(V.data_ptr<at::Half>());

    __half* O_ptr =
        reinterpret_cast<__half*>(O.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ----------------------------------------------------------------------------
    // Launch Kernel
    // ----------------------------------------------------------------------------

    int8_attention_general_kernel<<<grid, block, shmem, stream>>>(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        timestep_scales_ptr,
        static_cast<int>(timestep),
        B, H, N, D
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;
}

// ----------------------------------------------------------------------------
// PyBind
// ----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward",
          &int8_attention_forward,
          "General INT8 diffusion attention (CUDA)");
}