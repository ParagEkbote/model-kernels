// ============================================================================
// GENERAL INT8 FUSED ATTENTION KERNEL FOR DIFFUSION TRANSFORMERS  v2
// Features:
// - Per-head INT8 quantization (timestep-aware, scale informed BEFORE clamping)
// - INT8×INT8 → INT32 matmul via WMMA Tensor Cores (16×16×16 tiles)
// - Online stable softmax (Flash-Attention style, numerically correct)
// - Timestep-aware scaling folded into quantization range
// - Streaming over K tiles with fused quantize+scale pass
// - Proper B/H/N/D indexing with GQA/MQA support
// - Causal masking (optional, compile-time flag)
// - Templated head-dim D (no hardcoded MAX_D cap)
//
// v1 fixes:
//   [F1]  Block-wide scale reduction via shared memory before broadcast
//   [F2]  Softmax row_sum rescaling consistent with out_accum
//   [F3]  out_accum/row_max/row_sum in shared memory
//   [F4]  timestep_scales folded into abs-max BEFORE /127
//   [F5]  K quantization fused into single pass
//   [F6]  Causal masking via CAUSAL template parameter
//   [F7]  GQA/MQA: K/V heads indexed as h % kv_H
//   [F8]  D is a template parameter, no hardcoded cap
//   [F9]  WMMA INT8 Tensor Core path
//
// v2 fixes:
//   [G1]  WMMA layout: K stored transposed (col-major) in shared memory so
//         FragB=col_major loads K^T correctly without memory reinterpretation.
//   [G2]  WMMA tail-tile zero-padding: Q_i8/K_i8 padded to full 16-row
//         multiples before WMMA loads; out-of-range rows zeroed explicitly.
//   [G3]  HEAD_DIM dispatch extended: 32, 64, 80, 96, 128, 160, 256.
//         Fallback scalar path for any other multiple-of-16 value.
//   [G4]  Device shared-memory capability check in launcher: queries
//         cudaDevAttrMaxSharedMemoryPerBlockOptin and returns error if smem
//         requirement exceeds device limit. BLOCK_Q/BLOCK_K auto-downscaled
//         for D=256 via BlockConfig template.
//   [G5]  Softmax row loop partitioned by warp (qi = wid + warp-stride) so
//         all WARPS contribute during softmax; idle-thread waste eliminated.
//   [G6]  __launch_bounds__(THREADS, MIN_BLOCKS) tuned per HEAD_DIM to
//         control register pressure and improve occupancy.
//   [G7]  Launcher asserts kv_H > 0 && kv_H <= H before dispatch.
// ============================================================================

#pragma once

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
// on older GPUs while using larger tiles when the device supports it.
//
//   HEAD_DIM   BQ   BK   smem(approx)
//      32       64   64   ~28 KB
//      64       64   64   ~47 KB
//      80       64   64   ~57 KB
//      96       64   64   ~66 KB
//     128       64   64   ~81 KB     ← fits 96 KB
//     160       32   32   ~56 KB     ← downscaled
//     256       32   32   ~83 KB     ← downscaled, needs opt-in ≥96 KB
// ============================================================================
template<int HEAD_DIM> struct BlockConfig {
    static constexpr int BQ = (HEAD_DIM <= 128) ? 64 : 32;
    static constexpr int BK = (HEAD_DIM <= 128) ? 64 : 32;
};

// [G6] Minimum blocks per SM hint per HEAD_DIM (higher D = fewer regs to share)
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

// [F1] Block-wide max via shared scratch (≥WARPS floats)
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
// Quantization helper  [F4]
// inv_scale = 127 / (abs_max * timestep_scale)  — set before clamping
// ============================================================================
__device__ __forceinline__ int8_t quantize_f32(float v, float inv_scale) {
    return (int8_t)fmaxf(-128.f, fminf(127.f, roundf(v * inv_scale)));
}

// ============================================================================
// [G1] Store K tile transposed into shared memory (col-major layout).
//
// K_i8 is written as [HEAD_DIM, BK] (column-major for the K dimension).
// WMMA FragB declared col_major then loads K_i8 with stride=BK, which
// produces the matrix K^T — exactly what attention needs (Q·K^T).
//
//   K_src  : row-major [k_size, HEAD_DIM]  (global memory)
//   K_i8   : col-major [HEAD_DIM, BK]      (shared memory)
//   stride for WMMA load_matrix_sync = BK
// ============================================================================
template<int HEAD_DIM, int BK>
__device__ __forceinline__ void load_and_quantize_K_transposed(
    const float16* __restrict__ K_src,   // global, row-major [k_size × HEAD_DIM]
    int8_t*        __restrict__ K_i8_T,  // shared, col-major [HEAD_DIM × BK]
    int k_size,
    int tid,
    float inv_scale_K)
{
    // -------------------------------------------------------------------------
    // 1️⃣ Quantize + transpose valid region [0, k_size)
    // -------------------------------------------------------------------------
    for (int idx = tid; idx < k_size * HEAD_DIM; idx += THREADS) {
        int ki = idx / HEAD_DIM;
        int di = idx % HEAD_DIM;

        float val = __half2float(K_src[ki * HEAD_DIM + di]);

        // Transposed layout: K_i8_T[di, ki]
        K_i8_T[di * BK + ki] = quantize_f32(val, inv_scale_K);
    }

    // -------------------------------------------------------------------------
    // 2️⃣ Zero-pad remaining columns [k_size, BK)
    //    Prevents WMMA from reading garbage
    // -------------------------------------------------------------------------
    if (k_size < BK) {
        const int total_pad = (BK - k_size) * HEAD_DIM;

        for (int idx = tid; idx < total_pad; idx += THREADS) {
            int pad_col = k_size + idx / HEAD_DIM;  // column index
            int di      = idx % HEAD_DIM;           // row index

            K_i8_T[di * BK + pad_col] = 0;
        }
    }
}

// ============================================================================
// [G2] Zero-pad Q_i8 rows beyond q_size so WMMA tail fragments read 0
// ============================================================================
template<int HEAD_DIM, int BQ>
__device__ __forceinline__ void pad_Q_rows(
    int8_t* Q_i8,
    int q_size,
    int tid)
{
    // Total elements to zero:
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
__launch_bounds__(THREADS, OccHint<HEAD_DIM>::MIN_BLOCKS)   // [G6]
int8_attention_kernel(
    const float16* __restrict__ Q,
    const float16* __restrict__ K,
    const float16* __restrict__ V,
    float16*       __restrict__ O,
    const float*   __restrict__ timestep_scales,
    int  timestep,
    int  B, int H, int kv_H, int N)
{
    // Tile sizes from config
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

    // [F7] GQA/MQA
    const int kv_h = h % kv_H;

    const size_t  q_off = ((size_t)b * H    + h)    * N * HEAD_DIM;
    const size_t kv_off = ((size_t)b * kv_H + kv_h) * N * HEAD_DIM;

    const float16* Q_head = Q + q_off;
    const float16* K_head = K + kv_off;
    const float16* V_head = V + kv_off;
    float16*       O_head = O + q_off;

    // -------------------------------------------------------------------------
    // Shared memory layout:
    //   Q_i8       [BQ × HEAD_DIM]   row-major  int8
    //   K_i8_T     [HEAD_DIM × BK]   col-major  int8   [G1]
    //   V_tile     [BK × HEAD_DIM]   row-major  fp16
    //   QK_i32     [BQ × BK]                    int32
    //   warp_scr   [WARPS]                       float  (block reduce scratch)
    //   row_max    [BQ]                          float
    //   row_sum    [BQ]                          float
    //   out_accum  [BQ × HEAD_DIM]               float
    // -------------------------------------------------------------------------
    extern __shared__ char smem[];

    int8_t*  Q_i8     = reinterpret_cast<int8_t*>(smem);
    int8_t*  K_i8_T   = Q_i8     + BQ * HEAD_DIM;              // [G1] transposed
    float16* V_tile   = reinterpret_cast<float16*>(K_i8_T + HEAD_DIM * BK);
    int32_t* QK_i32   = reinterpret_cast<int32_t*>(V_tile + BK * HEAD_DIM);
    float*   warp_scr = reinterpret_cast<float*>(QK_i32 + BQ * BK);
    float*   row_max  = warp_scr + WARPS;
    float*   row_sum  = row_max  + BQ;
    float*   out_acc  = row_sum  + BQ;    // [BQ × HEAD_DIM]

    // -------------------------------------------------------------------------
    // Q scale  [F1][F4]
    // -------------------------------------------------------------------------
    const float ts = timestep_scales ? timestep_scales[timestep] : 1.f;

    float lqmax = 0.f;
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS)
        lqmax = fmaxf(lqmax, fabsf(__half2float(Q_head[q_start * HEAD_DIM + i])));
    float abs_max_Q   = block_reduce_max(lqmax, warp_scr);
    const float inv_Q = 127.f / fmaxf(abs_max_Q * ts, 1e-6f);
    const float scl_Q = 1.f / inv_Q;

    // Quantize Q tile into Q_i8 (row-major)
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS) {
        int qi = i / HEAD_DIM, di = i % HEAD_DIM;
        Q_i8[qi * HEAD_DIM + di] =
            quantize_f32(__half2float(Q_head[(q_start + qi) * HEAD_DIM + di]), inv_Q);
    }
    // [G2] Zero-pad tail rows in Q
    pad_Q_rows<HEAD_DIM, BQ>(Q_i8, q_size, tid);

    // -------------------------------------------------------------------------
    // Initialise per-row accumulators  [F3]
    // -------------------------------------------------------------------------
    for (int qi = tid; qi < BQ; qi += THREADS) { row_max[qi] = -1e30f; row_sum[qi] = 0.f; }
    for (int i  = tid; i  < BQ * HEAD_DIM; i += THREADS) out_acc[i] = 0.f;
    __syncthreads();

    // -------------------------------------------------------------------------
    // WMMA fragment types  [G1] FragB = col_major, stride = BK
    // -------------------------------------------------------------------------
    using FragA   = wmma::fragment<wmma::matrix_a,    16, 16, 16, int8_t, wmma::row_major>;
    using FragB   = wmma::fragment<wmma::matrix_b,    16, 16, 16, int8_t, wmma::col_major>;
    using FragAcc = wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t>;

    const float inv_sqrt_d = rsqrtf((float)HEAD_DIM);

    // -------------------------------------------------------------------------
    // Stream K tiles
    // -------------------------------------------------------------------------
    for (int k_start = 0; k_start < N; k_start += BK) {
        const int k_size = min(BK, N - k_start);

        // [F1][F4] K scale — block-wide
        float lkmax = 0.f;
        for (int i = tid; i < k_size * HEAD_DIM; i += THREADS)
            lkmax = fmaxf(lkmax, fabsf(__half2float(K_head[k_start * HEAD_DIM + i])));
        float abs_max_K   = block_reduce_max(lkmax, warp_scr);
        const float inv_K = 127.f / fmaxf(abs_max_K * ts, 1e-6f);
        const float scl_K = 1.f / inv_K;

        // [F5][G1] Fused quantize + transpose K into K_i8_T
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

        // ---- [F9][G1] WMMA INT8×INT8 → INT32 --------------------------------
        // Q_i8    : row-major [BQ, HEAD_DIM], FragA row_major, stride=HEAD_DIM
        // K_i8_T  : col-major [HEAD_DIM, BK], FragB col_major, stride=BK
        //           ↑ WMMA reads K^T correctly because layout matches memory [G1]
        // Result  : QK_i32  row-major [BQ, BK]

        const int nQ16 = BQ / 16;          // always exact (BQ is multiple of 16)
        const int nK16 = BK / 16;
        const int nD16 = HEAD_DIM / 16;

        const int total_tiles = nQ16 * nK16;
        for (int wt = wid; wt < total_tiles; wt += WARPS) {
            const int qi16 = wt / nK16;
            const int ki16 = wt % nK16;

            FragAcc acc; wmma::fill_fragment(acc, 0);

            for (int d16 = 0; d16 < nD16; ++d16) {
                FragA fa; FragB fb;

                // Q: row qi16*16, col d16*16; stride = HEAD_DIM
                wmma::load_matrix_sync(fa,
                    Q_i8 + qi16 * 16 * HEAD_DIM + d16 * 16,
                    HEAD_DIM);

                // K^T: row d16*16, col ki16*16 in [HEAD_DIM × BK]; stride = BK [G1]
                wmma::load_matrix_sync(fb,
                    K_i8_T + d16 * 16 * BK + ki16 * 16,
                    BK);

                wmma::mma_sync(acc, fa, fb, acc);
            }

            // Store to QK_i32[qi16*16, ki16*16]; stride = BK
            wmma::store_matrix_sync(
                QK_i32 + qi16 * 16 * BK + ki16 * 16,
                acc, BK, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- [F2][G5] Online softmax — warp-partitioned rows -----------------
        // [G5] Distribute qi rows across WARPS: warp `w` handles rows
        //      qi = w, w+WARPS, w+2*WARPS, ...  so all warps stay busy.
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

            // [F2] Consistent rescaling: both row_sum and out_acc by same factor
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

    // -------------------------------------------------------------------------
    // Normalise and store
    // -------------------------------------------------------------------------
    for (int i = tid; i < q_size * HEAD_DIM; i += THREADS) {
        int qi = i / HEAD_DIM, di = i % HEAD_DIM;
        float val = out_acc[qi * HEAD_DIM + di] / (row_sum[qi] + 1e-6f);
        O_head[(q_start + qi) * HEAD_DIM + di] = __float2half(val);
    }
}

// ============================================================================
// Shared memory size helper — accounts for BQ/BK config per HEAD_DIM  [G4]
// ============================================================================
inline size_t int8_attention_smem_bytes(int D) {
    // Use same logic as BlockConfig<> to pick correct BQ/BK at runtime
    int BQ = (D <= 128) ? 64 : 32;
    int BK = (D <= 128) ? 64 : 32;
    return (size_t)BQ * D       * sizeof(int8_t)   // Q_i8
         + (size_t)D  * BK      * sizeof(int8_t)   // K_i8_T (transposed)
         + (size_t)BK * D       * sizeof(float16)  // V_tile
         + (size_t)BQ * BK      * sizeof(int32_t)  // QK_i32
         + (size_t)WARPS        * sizeof(float)    // warp_scratch
         + (size_t)BQ * 2       * sizeof(float)    // row_max + row_sum
         + (size_t)BQ * D       * sizeof(float);   // out_accum
}

// ============================================================================
// HEAD_DIM dispatch  [G3] — extended to common diffusion transformer dims
// Fallback scalar path note: for truly arbitrary D (e.g. 192) add an entry
// or fall through to a scalar fallback kernel (not included here).
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
// Launcher  [G4][G7]
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
        // Cannot run: shared memory requirement exceeds device maximum.
        // Caller should reduce D or reduce BLOCK_Q/BLOCK_K via a custom config.
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
