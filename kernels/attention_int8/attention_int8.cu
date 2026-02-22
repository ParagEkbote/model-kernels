// ============================================================================
// GENERAL INT8 FUSED ATTENTION KERNEL FOR DIFFUSION TRANSFORMERS
// Features:
// - Per-head INT8 quantization
// - INT8×INT8 → INT32 matmul
// - Online stable softmax
// - Timestep-aware scaling
// - Streaming over K tiles
// - Proper B/H/N/D indexing
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;
using float16 = __half;

#define BLOCK_Q 64
#define BLOCK_K 64
#define THREADS 256
#define MAX_D 256

// ----------------------------------------------------------------------------
// Warp Reductions
// ----------------------------------------------------------------------------

__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ----------------------------------------------------------------------------
// Quantization
// ----------------------------------------------------------------------------

__device__ __forceinline__
int8_t quantize(float16 val, float scale) {
    float f = __half2float(val) / scale;
    f = roundf(f);
    f = fmaxf(-128.f, fminf(127.f, f));
    return (int8_t)f;
}

// ----------------------------------------------------------------------------
// Main Kernel
// ----------------------------------------------------------------------------

extern "C"
__global__ void int8_attention_general_kernel(
    const float16* __restrict__ Q,
    const float16* __restrict__ K,
    const float16* __restrict__ V,
    float16* __restrict__ O,
    const float* __restrict__ timestep_scales,
    int timestep,
    int B, int H, int N, int D
) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_tile = blockIdx.z;

    int tid = threadIdx.x;

    int q_start = q_tile * BLOCK_Q;
    if (q_start >= N) return;

    int q_size = min(BLOCK_Q, N - q_start);

    // Offsets
    size_t head_offset = ((size_t)b * H + h) * N * D;

    const float16* Q_head = Q + head_offset;
    const float16* K_head = K + head_offset;
    const float16* V_head = V + head_offset;
    float16* O_head = O + head_offset;

    // Shared memory
    extern __shared__ char smem[];
    int8_t* Q_i8 = (int8_t*)smem;
    int8_t* K_i8 = Q_i8 + BLOCK_Q * MAX_D;
    float16* V_tile = (float16*)(K_i8 + BLOCK_K * MAX_D);
    int32_t* QK_i32 = (int32_t*)(V_tile + BLOCK_K * MAX_D);

    // ----------------------------------------------------------------------------
    // Stage 1: Compute Q scale (per head)
    // ----------------------------------------------------------------------------

    float local_max = 0.f;
    for (int i = tid; i < N * D; i += THREADS) {
        float val = fabsf(__half2float(Q_head[i]));
        local_max = max(local_max, val);
    }
    float scale_Q = warp_max(local_max);
    if (threadIdx.x == 0)
        scale_Q = max(1e-6f, scale_Q) / 127.f;

    scale_Q = __shfl_sync(0xffffffff, scale_Q, 0);

    if (timestep_scales)
        scale_Q *= timestep_scales[timestep];

    // ----------------------------------------------------------------------------
    // Load & Quantize Q Tile
    // ----------------------------------------------------------------------------

    for (int i = tid; i < q_size * D; i += THREADS) {
        int qi = i / D;
        int di = i % D;
        Q_i8[qi * D + di] =
            quantize(Q_head[(q_start + qi) * D + di], scale_Q);
    }
    __syncthreads();

    // ----------------------------------------------------------------------------
    // Online Softmax Variables
    // ----------------------------------------------------------------------------

    float row_max[BLOCK_Q];
    float row_sum[BLOCK_Q];

    for (int i = 0; i < q_size; ++i) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.f;
    }

    float out_accum[BLOCK_Q][MAX_D];

    for (int qi = 0; qi < q_size; ++qi)
        for (int di = 0; di < D; ++di)
            out_accum[qi][di] = 0.f;

    // ----------------------------------------------------------------------------
    // Stream over K tiles
    // ----------------------------------------------------------------------------

    for (int k_start = 0; k_start < N; k_start += BLOCK_K) {

        int k_size = min(BLOCK_K, N - k_start);

        // ---- Compute K scale ----
        float local_kmax = 0.f;
        for (int i = tid; i < k_size * D; i += THREADS) {
            float val = fabsf(__half2float(K_head[(k_start * D) + i]));
            local_kmax = max(local_kmax, val);
        }

        float scale_K = warp_max(local_kmax);
        if (threadIdx.x == 0)
            scale_K = max(1e-6f, scale_K) / 127.f;

        scale_K = __shfl_sync(0xffffffff, scale_K, 0);

        if (timestep_scales)
            scale_K *= timestep_scales[timestep];

        // ---- Load & Quantize K ----
        for (int i = tid; i < k_size * D; i += THREADS) {
            int ki = i / D;
            int di = i % D;
            K_i8[ki * D + di] =
                quantize(K_head[(k_start + ki) * D + di], scale_K);
        }

        // ---- Load V ----
        for (int i = tid; i < k_size * D; i += THREADS) {
            int ki = i / D;
            int di = i % D;
            V_tile[ki * D + di] =
                V_head[(k_start + ki) * D + di];
        }

        __syncthreads();

        // ---- INT8 Matmul (naive path; WMMA path can replace) ----

        for (int idx = tid; idx < q_size * k_size; idx += THREADS) {
            int qi = idx / k_size;
            int ki = idx % k_size;

            int32_t acc = 0;
            for (int d = 0; d < D; ++d)
                acc += (int32_t)Q_i8[qi * D + d] *
                       (int32_t)K_i8[ki * D + d];

            QK_i32[qi * k_size + ki] = acc;
        }

        __syncthreads();

        // ---- Online Softmax Update ----

        float inv_sqrt_d = rsqrtf((float)D);
        float combined_scale = inv_sqrt_d / (scale_Q * scale_K);

        for (int qi = tid; qi < q_size; qi += THREADS) {

            float tile_max = -1e30f;

            for (int ki = 0; ki < k_size; ++ki) {
                float val =
                    (float)QK_i32[qi * k_size + ki] * combined_scale;
                tile_max = max(tile_max, val);
            }

            float new_max = max(row_max[qi], tile_max);
            float exp_scale_old = expf(row_max[qi] - new_max);
            float exp_scale_new = expf(tile_max - new_max);

            float tile_sum = 0.f;

            for (int ki = 0; ki < k_size; ++ki) {
                float val =
                    (float)QK_i32[qi * k_size + ki] * combined_scale;
                float w = expf(val - new_max);
                tile_sum += w;

                for (int d = 0; d < D; ++d)
                    out_accum[qi][d] =
                        out_accum[qi][d] * exp_scale_old +
                        w * __half2float(V_tile[ki * D + d]);
            }

            row_sum[qi] =
                row_sum[qi] * exp_scale_old +
                tile_sum * exp_scale_new;

            row_max[qi] = new_max;
        }

        __syncthreads();
    }

    // ----------------------------------------------------------------------------
    // Final Normalize & Store
    // ----------------------------------------------------------------------------

    for (int i = tid; i < q_size * D; i += THREADS) {
        int qi = i / D;
        int di = i % D;

        float val = out_accum[qi][di] / (row_sum[qi] + 1e-6f);
        O_head[(q_start + qi) * D + di] = __float2half(val);
    }
}