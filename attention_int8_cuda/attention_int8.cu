// ============================================================================
// INT8 FUSED ATTENTION KERNEL FOR DIFFUSION TRANSFORMERS - PyTorch Integration
// v2 with comprehensive safeguards and PyTorch tensor validation
// ============================================================================

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <algorithm>

using namespace nvcuda;
using float16 = __half;

// ============================================================================
// Block configuration: tile sizes vary by HEAD_DIM to stay within 96 KB shmem
// ============================================================================
template<int HEAD_DIM> struct BlockConfig {
    static constexpr int BQ = (HEAD_DIM <= 128) ? 64 : 32;
    static constexpr int BK = (HEAD_DIM <= 128) ? 64 : 32;
};

// [G6] Minimum blocks per SM hint per HEAD_DIM
template<int HEAD_DIM> struct OccHint {
    static constexpr int MIN_BLOCKS = (HEAD_DIM <= 64) ? 2 : 1;
};

#define THREADS 128
#define WARPS   (THREADS / 32)

// ============================================================================
// Warp/block reductions
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem_scratch) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem_scratch[wid] = val;
    __syncthreads();
    float res = (threadIdx.x < WARPS) ? smem_scratch[threadIdx.x] : -1e30f;
    if (wid == 0) res = warp_reduce_max(res);
    __syncthreads();
    if (threadIdx.x == 0) smem_scratch[0] = res;
    __syncthreads();
    return smem_scratch[0];
}

// ============================================================================
// Quantization helper [F4]
// ============================================================================
__device__ __forceinline__ int8_t quantize_f32(float v, float inv_scale) {
    return (int8_t)fmaxf(-128.f, fminf(127.f, roundf(v * inv_scale)));
}

// ============================================================================
// [G1] Store K tile transposed into shared memory (col-major layout)
// ============================================================================
template<int HEAD_DIM, int BK>
__device__ __forceinline__ void load_and_quantize_K_transposed(
    const float16* __restrict__ K_src,
    int8_t*        __restrict__ K_i8_T,
    int k_size,
    int tid,
    float inv_scale_K)
{
    // Quantize + transpose valid region
    for (int idx = tid; idx < k_size * HEAD_DIM; idx += THREADS) {
        int ki = idx / HEAD_DIM;
        int di = idx % HEAD_DIM;
        float val = __half2float(K_src[ki * HEAD_DIM + di]);
        K_i8_T[di * BK + ki] = quantize_f32(val, inv_scale_K);
    }

    // Zero-pad remaining columns [k_size, BK)
    if (k_size < BK) {
        const int total_pad = (BK - k_size) * HEAD_DIM;
        for (int idx = tid; idx < total_pad; idx += THREADS) {
            int pad_col = k_size + idx / HEAD_DIM;
            int di      = idx % HEAD_DIM;
            K_i8_T[di * BK + pad_col] = 0;
        }
    }
}

// ============================================================================
// [G2] Zero-pad Q_i8 rows beyond q_size
// ============================================================================
template<int HEAD_DIM, int BQ>
__device__ __forceinline__ void pad_Q_rows(
    int8_t* Q_i8,
    int q_size,
    int tid)
{
    const int total_pad_elems = (BQ - q_size) * HEAD_DIM;
    for (int idx = tid; idx < total_pad_elems; idx += THREADS) {
        int qi = q_size + idx / HEAD_DIM;
        int di = idx % HEAD_DIM;
        Q_i8[qi * HEAD_DIM + di] = 0;
    }
}

// ============================================================================
// Main kernel
// ============================================================================
template<int HEAD_DIM, bool CAUSAL>
__global__ void
__launch_bounds__(THREADS, OccHint<HEAD_DIM>::MIN_BLOCKS)
int8_attention_kernel(
    const float16* __restrict__ Q,
    const float16* __restrict__ K,
    const float16* __restrict__ V,
    float16*       __restrict__ O,
    const float*   __restrict__ timestep_scales,
    int  timestep,
    int  B, int H, int kv_H, int N)
{
    constexpr int BQ = BlockConfig<HEAD_DIM>::BQ;
    constexpr int BK = BlockConfig<HEAD_DIM>::BK;

    static_assert(BQ % 16 == 0, "BQ must be multiple of 16");
    static_assert(BK % 16 == 0, "BK must be multiple of 16");
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16");

    const int b      = blockIdx.x;
    const int h      = blockIdx.y;
    const int q_tile = blockIdx.z;
    const int tid    = threadIdx.x;
    const int wid    = tid >> 5;
    const int lane   = tid & 31;

    const int q_start = q_tile * BQ;
    if (q_start >= N) return;
    const int q_size  = min(BQ, N - q_start);

    const int kv_h = h % kv_H;

    const size_t  q_off = ((size_t)b * H    + h)    * N * HEAD_DIM;
    const size_t kv_off = ((size_t)b * kv_H + kv_h) * N * HEAD_DIM;

    const float16* Q_head = Q + q_off;
    const float16* K_head = K + kv_off;
    const float16* V_head = V + kv_off;
    float16*       O_head = O + q_off;

    // Shared memory layout
    extern __shared__ char smem[];

    int8_t*  Q_i8     = reinterpret_cast<int8_t*>(smem);
    int8_t*  K_i8_T   = Q_i8     + BQ * HEAD_DIM;
    float16* V_tile   = reinterpret_cast<float16*>(K_i8_T + HEAD_DIM * BK);
    int32_t* QK_i32   = reinterpret_cast<int32_t*>(V_tile + BK * HEAD_DIM);
    float*   warp_scr = reinterpret_cast<float*>(QK_i32 + BQ * BK);
    float*   row_max  = warp_scr + WARPS;
    float*   row_sum  = row_max  + BQ;
    float*   out_acc  = row_sum  + BQ;

    // Q scale [F1][F4]
    const float ts = timestep_scales ? timestep_scales[timestep] : 1.f;

    float lqmax = 0.f;
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS)
        lqmax = fmaxf(lqmax, fabsf(__half2float(Q_head[q_start * HEAD_DIM + i])));
    float abs_max_Q   = block_reduce_max(lqmax, warp_scr);
    const float inv_Q = 127.f / fmaxf(abs_max_Q * ts, 1e-6f);
    const float scl_Q = 1.f / inv_Q;

    // Quantize Q tile
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS) {
        int qi = i / HEAD_DIM, di = i % HEAD_DIM;
        Q_i8[qi * HEAD_DIM + di] =
            quantize_f32(__half2float(Q_head[(q_start + qi) * HEAD_DIM + di]), inv_Q);
    }
    pad_Q_rows<HEAD_DIM, BQ>(Q_i8, q_size, tid);

    // Initialise per-row accumulators [F3]
    for (int qi = tid; qi < BQ; qi += THREADS) { row_max[qi] = -1e30f; row_sum[qi] = 0.f; }
    for (int i  = tid; i  < BQ * HEAD_DIM; i += THREADS) out_acc[i] = 0.f;
    __syncthreads();

    // WMMA fragment types [G1]
    using FragA   = wmma::fragment<wmma::matrix_a,    16, 16, 16, int8_t, wmma::row_major>;
    using FragB   = wmma::fragment<wmma::matrix_b,    16, 16, 16, int8_t, wmma::col_major>;
    using FragAcc = wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t>;

    const float inv_sqrt_d = rsqrtf((float)HEAD_DIM);

    // Stream K tiles
    for (int k_start = 0; k_start < N; k_start += BK) {
        const int k_size = min(BK, N - k_start);

        float lkmax = 0.f;
        for (int i = tid; i < k_size * HEAD_DIM; i += THREADS)
            lkmax = fmaxf(lkmax, fabsf(__half2float(K_head[k_start * HEAD_DIM + i])));
        float abs_max_K   = block_reduce_max(lkmax, warp_scr);
        const float inv_K = 127.f / fmaxf(abs_max_K * ts, 1e-6f);
        const float scl_K = 1.f / inv_K;

        // [F5][G1] Fused quantize + transpose K
        load_and_quantize_K_transposed<HEAD_DIM, BK>(
            K_head + k_start * HEAD_DIM, K_i8_T, k_size, tid, inv_K);

        // Load V tile
        for (int i = tid; i < k_size * HEAD_DIM; i += THREADS) {
            int ki = i / HEAD_DIM, di = i % HEAD_DIM;
            V_tile[ki * HEAD_DIM + di] = V_head[(k_start + ki) * HEAD_DIM + di];
        }

        // Zero QK_i32 accumulator
        for (int i = tid; i < BQ * BK; i += THREADS) QK_i32[i] = 0;
        __syncthreads();

        // [F9][G1] WMMA INT8×INT8 → INT32
        const int nQ16 = BQ / 16;
        const int nK16 = BK / 16;
        const int nD16 = HEAD_DIM / 16;

        const int total_tiles = nQ16 * nK16;
        for (int wt = wid; wt < total_tiles; wt += WARPS) {
            const int qi16 = wt / nK16;
            const int ki16 = wt % nK16;

            FragAcc acc; wmma::fill_fragment(acc, 0);

            for (int d16 = 0; d16 < nD16; ++d16) {
                FragA fa; FragB fb;

                wmma::load_matrix_sync(fa,
                    Q_i8 + qi16 * 16 * HEAD_DIM + d16 * 16,
                    HEAD_DIM);

                wmma::load_matrix_sync(fb,
                    K_i8_T + d16 * 16 * BK + ki16 * 16,
                    BK);

                wmma::mma_sync(acc, fa, fb, acc);
            }

            wmma::store_matrix_sync(
                QK_i32 + qi16 * 16 * BK + ki16 * 16,
                acc, BK, wmma::mem_row_major);
        }
        __syncthreads();

        // [F2][G5] Online softmax — warp-partitioned rows
        const float cscale = inv_sqrt_d * scl_Q * scl_K;

        for (int qi = wid; qi < q_size; qi += WARPS) {
            const int q_global = q_start + qi;

            float tile_max = -1e30f;
            for (int ki = 0; ki < k_size; ++ki) {
                if (CAUSAL && (k_start + ki) > q_global) continue;
                float s = (float)QK_i32[qi * BK + ki] * cscale;
                tile_max = fmaxf(tile_max, s);
            }

            const float old_max = row_max[qi];
            const float new_max = fmaxf(old_max, tile_max);
            const float rescale = expf(old_max - new_max);

            // [F2] Consistent rescaling
            row_sum[qi] *= rescale;
            for (int d = 0; d < HEAD_DIM; ++d)
                out_acc[qi * HEAD_DIM + d] *= rescale;

            float tsum = 0.f;
            for (int ki = 0; ki < k_size; ++ki) {
                if (CAUSAL && (k_start + ki) > q_global) continue;
                float s = (float)QK_i32[qi * BK + ki] * cscale;
                float w = expf(s - new_max);
                tsum   += w;
                for (int d = 0; d < HEAD_DIM; ++d)
                    out_acc[qi * HEAD_DIM + d] += w * __half2float(V_tile[ki * HEAD_DIM + d]);
            }
            row_sum[qi] += tsum;
            row_max[qi]  = new_max;
        }
        __syncthreads();
    }

    // Normalise and store
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS) {
        int qi = i / HEAD_DIM, di = i % HEAD_DIM;
        float val = out_acc[qi * HEAD_DIM + di] / (row_sum[qi] + 1e-6f);
        O_head[(q_start + qi) * HEAD_DIM + di] = __float2half(val);
    }
}

// ============================================================================
// Shared memory size helper [G4]
// ============================================================================
inline size_t int8_attention_smem_bytes(int D) {
    int BQ = (D <= 128) ? 64 : 32;
    int BK = (D <= 128) ? 64 : 32;
    return (size_t)BQ * D       * sizeof(int8_t)
         + (size_t)D  * BK      * sizeof(int8_t)
         + (size_t)BK * D       * sizeof(float16)
         + (size_t)BQ * BK      * sizeof(int32_t)
         + (size_t)WARPS        * sizeof(float)
         + (size_t)BQ * 2       * sizeof(float)
         + (size_t)BQ * D       * sizeof(float);
}

// ============================================================================
// HEAD_DIM dispatch [G3]
// ============================================================================
#define DISPATCH_HD(D, CAUSAL, BODY)               \
    if      (D ==  32) { constexpr int HD =  32; BODY; } \
    else if (D ==  64) { constexpr int HD =  64; BODY; } \
    else if (D ==  80) { constexpr int HD =  80; BODY; } \
    else if (D ==  96) { constexpr int HD =  96; BODY; } \
    else if (D == 128) { constexpr int HD = 128; BODY; } \
    else if (D == 160) { constexpr int HD = 160; BODY; } \
    else if (D == 256) { constexpr int HD = 256; BODY; } \
    else { return cudaErrorInvalidValue; }

// ============================================================================
// Launcher [G4][G7]
// ============================================================================
inline cudaError_t launch_int8_attention(
    const float16* Q,
    const float16* K,
    const float16* V,
    float16*       O,
    const float*   timestep_scales,
    int  timestep,
    int  B, int H, int kv_H, int N, int D,
    bool causal,
    cudaStream_t stream = 0)
{
    // [G7] Guard: kv_H must be valid
    if (kv_H <= 0 || kv_H > H) return cudaErrorInvalidValue;
    if (D % 16 != 0)           return cudaErrorInvalidValue;

    // [G4] Query device shared-memory limit
    int dev = 0;
    cudaGetDevice(&dev);
    int max_smem_optin = 0;
    cudaDeviceGetAttribute(&max_smem_optin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    const size_t smem = int8_attention_smem_bytes(D);
    if ((int)smem > max_smem_optin) {
        return cudaErrorMemoryAllocation;
    }

    int BQ = (D <= 128) ? 64 : 32;
    const dim3 grid(B, H, (N + BQ - 1) / BQ);
    const dim3 block(THREADS);

#define LAUNCH_KERNEL(HD, CAU)                                                      \
    do {                                                                            \
        cudaError_t _e = cudaFuncSetAttribute(                                      \
            int8_attention_kernel<HD, CAU>,                                         \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);               \
        if (_e != cudaSuccess) return _e;                                           \
        int8_attention_kernel<HD, CAU><<<grid, block, smem, stream>>>(             \
            Q, K, V, O, timestep_scales, timestep, B, H, kv_H, N);               \
    } while(0)

    if (causal) { DISPATCH_HD(D, true,  LAUNCH_KERNEL(HD, true)); }
    else        { DISPATCH_HD(D, false, LAUNCH_KERNEL(HD, false)); }

#undef LAUNCH_KERNEL
    return cudaGetLastError();
}

// ============================================================================
// PyTorch Binding with Comprehensive Safeguards
// ============================================================================

void int8_attention_check_inputs(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor timestep_scales)
{
    // Device checks
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.device().is_cuda(), "O must be a CUDA tensor");
    TORCH_CHECK(Q.device() == K.device(), "Q and K must be on the same device");
    TORCH_CHECK(Q.device() == V.device(), "Q and V must be on the same device");
    TORCH_CHECK(Q.device() == O.device(), "Q and O must be on the same device");

    if (timestep_scales.defined()) {
        TORCH_CHECK(timestep_scales.device().is_cuda(),
                    "timestep_scales must be a CUDA tensor if provided");
        TORCH_CHECK(timestep_scales.device() == Q.device(),
                    "timestep_scales must be on the same device as Q");
    }

    // Contiguity checks
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");

    // Shape checks: [B, H, N, D]
    TORCH_CHECK(Q.dim() == 4, "Q must have 4 dimensions [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must have 4 dimensions [B, kv_H, N, D]");
    TORCH_CHECK(V.dim() == 4, "V must have 4 dimensions [B, kv_H, N, D]");
    TORCH_CHECK(O.dim() == 4, "O must have 4 dimensions [B, H, N, D]");

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    int kv_H = K.size(1);

    TORCH_CHECK(K.size(0) == B, "K batch dimension must match Q");
    TORCH_CHECK(V.size(0) == B, "V batch dimension must match Q");
    TORCH_CHECK(O.size(0) == B, "O batch dimension must match Q");

    TORCH_CHECK(K.size(2) == N, "K sequence length must match Q");
    TORCH_CHECK(V.size(2) == N, "V sequence length must match Q");
    TORCH_CHECK(O.size(2) == N, "O sequence length must match Q");

    TORCH_CHECK(K.size(3) == D, "K head dimension must match Q");
    TORCH_CHECK(V.size(3) == D, "V head dimension must match Q");
    TORCH_CHECK(O.size(3) == D, "O head dimension must match Q");

    TORCH_CHECK(Q.size(1) % kv_H == 0,
                "Q head dimension (H) must be divisible by K head dimension (kv_H)");

    // Dtype checks: accept fp16, fp32, or bfloat16
    TORCH_CHECK(Q.scalar_type() == at::ScalarType::Half ||
                Q.scalar_type() == at::ScalarType::Float ||
                Q.scalar_type() == at::ScalarType::BFloat16,
                "Q must be float16, float32, or bfloat16");

    TORCH_CHECK(Q.scalar_type() == K.scalar_type(),
                "Q and K must have the same dtype");
    TORCH_CHECK(Q.scalar_type() == V.scalar_type(),
                "Q and V must have the same dtype");
    TORCH_CHECK(Q.scalar_type() == O.scalar_type(),
                "Q and O must have the same dtype");

    // HEAD_DIM support checks
    TORCH_CHECK(D % 16 == 0, "HEAD_DIM must be a multiple of 16");
    TORCH_CHECK(D == 32 || D == 64 || D == 80 || D == 96 ||
                D == 128 || D == 160 || D == 256,
                "HEAD_DIM must be one of: 32, 64, 80, 96, 128, 160, 256");

    // kv_H validity [G7]
    TORCH_CHECK(kv_H > 0 && kv_H <= H,
                "kv_H must be > 0 and <= H");

    if (timestep_scales.defined()) {
        TORCH_CHECK(timestep_scales.dim() == 1,
                    "timestep_scales must be 1-dimensional");
        TORCH_CHECK(timestep_scales.scalar_type() == at::ScalarType::Float,
                    "timestep_scales must be float32");
    }
}

torch::Tensor int8_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor timestep_scales,
    int timestep,
    bool causal)
{
    int8_attention_check_inputs(Q, K, V, Q, timestep_scales);

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    int kv_H = K.size(1);

    torch::Tensor O = torch::zeros_like(Q);

    // Convert to float16 if needed
    torch::Tensor Q_fp16 = (Q.scalar_type() == at::ScalarType::Half) ?
                           Q : Q.to(at::ScalarType::Half);
    torch::Tensor K_fp16 = (K.scalar_type() == at::ScalarType::Half) ?
                           K : K.to(at::ScalarType::Half);
    torch::Tensor V_fp16 = (V.scalar_type() == at::ScalarType::Half) ?
                           V : V.to(at::ScalarType::Half);
    torch::Tensor O_fp16 = O.to(at::ScalarType::Half);

    const float* ts_ptr = timestep_scales.defined() ?
                          timestep_scales.data_ptr<float>() : nullptr;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(Q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = launch_int8_attention(
        reinterpret_cast<const float16*>(Q_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const float16*>(K_fp16.data_ptr<at::Half>()),
        reinterpret_cast<const float16*>(V_fp16.data_ptr<at::Half>()),
        reinterpret_cast<float16*>(O_fp16.data_ptr<at::Half>()),
        ts_ptr,
        timestep,
        B, H, kv_H, N, D,
        causal,
        stream);

    TORCH_CHECK(err == cudaSuccess,
                "int8_attention kernel launch failed: ",
                cudaGetErrorString(err));

    // Convert back to original dtype if needed
    if (Q.scalar_type() != at::ScalarType::Half) {
        O = O_fp16.to(Q.scalar_type());
    } else {
        O = O_fp16;
    }

    return O;
}

// ============================================================================
// TORCH_LIBRARY binding (if using PyTorch 1.9+)
// ============================================================================
TORCH_LIBRARY(int8_attn, m) {
    m.def("int8_attention(Tensor Q, Tensor K, Tensor V, Tensor? timestep_scales, int timestep, bool causal) -> Tensor",
          &int8_attention);
}
