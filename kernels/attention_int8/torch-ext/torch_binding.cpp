#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" __global__ void int8_attention_fused_kernel(
    const __half* Q_fp16,
    const __half* K_fp16,
    const __half* V_fp16,
    __half* output,
    const float* timestep_scales,
    int timestep,
    int N, int D,
    int BLOCK_Q, int BLOCK_K
);

static void check_inputs(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), "", name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kHalf, "", name, " must be torch.float16 (__half)");
    TORCH_CHECK(t.is_contiguous(), "", name, " must be contiguous");
}

torch::Tensor int8_attention_forward(
    torch::Tensor Q, // [B, H, N, D], fp16
    torch::Tensor K, // [B, H, N, D], fp16
    torch::Tensor V, // [B, H, N, D], fp16
    c10::optional<torch::Tensor> timestep_scales_opt,
    int64_t timestep
) {
    check_inputs(Q, "Q");
    check_inputs(K, "K");
    check_inputs(V, "V");

    TORCH_CHECK(Q.dim() == 4, "Q must be [B,H,N,D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D = Q.size(3);

    torch::Tensor out = torch::empty_like(Q);

    // Optional timestep scales (CPU or CUDA float tensor)
    const float* timestep_scales_ptr = nullptr;
    if (timestep_scales_opt.has_value()) {
        auto ts = timestep_scales_opt.value();
        TORCH_CHECK(ts.device().is_cuda(), "timestep_scales must be a CUDA tensor or None");
        TORCH_CHECK(ts.dtype() == torch::kFloat, "timestep_scales must be float32");
        timestep_scales_ptr = ts.data_ptr<float>();
    }

    // Launch configuration
    const int threads = 256;
    dim3 block(threads);
    dim3 grid((int)B, (int)H);

    // Match defaults in kernel
    const int BLOCK_Q = 64;
    const int BLOCK_K = 64;

    // Compute shared memory size conservatively (must match kernel's shared layout)
    // Q_tile_i8: BLOCK_Q * D * int8
    // K_tile_i8: BLOCK_K * D * int8
    // V_tile_fp16: BLOCK_K * D * __half
    // QK_accum_i32: BLOCK_Q * BLOCK_K * int32
    // logits: BLOCK_Q * N * __half
    // workspace: a small float buffer
    size_t shmem = 0;
    shmem += size_t(BLOCK_Q) * size_t(D) * sizeof(int8_t);
    shmem += size_t(BLOCK_K) * size_t(D) * sizeof(int8_t);
    shmem += size_t(BLOCK_K) * size_t(D) * sizeof(__half);
    shmem += size_t(BLOCK_Q) * size_t(BLOCK_K) * sizeof(int32_t);
    shmem += size_t(BLOCK_Q) * size_t(N) * sizeof(__half);
    shmem += 1024 * sizeof(float); // workspace headroom

    // Get stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Raw pointers
    const __half* Q_ptr = reinterpret_cast<const __half*>(Q.data_ptr<at::Half>());
    const __half* K_ptr = reinterpret_cast<const __half*>(K.data_ptr<at::Half>());
    const __half* V_ptr = reinterpret_cast<const __half*>(V.data_ptr<at::Half>());
    __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    // Launch kernel
    int8_attention_fused_kernel<<<grid, block, shmem, stream>>>(
        Q_ptr, K_ptr, V_ptr, out_ptr,
        timestep_scales_ptr,
        (int)timestep,
        (int)N, (int)D,
        BLOCK_Q, BLOCK_K
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_attention_forward", &int8_attention_forward, "INT8 fused attention forward (CUDA)");
}
