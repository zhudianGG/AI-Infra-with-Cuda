#include <cuda_fp16.h>
#include <float.h>
#include "ppl_common.cuh"

struct dynamic_batching_decoding_cache_attention_kernel_param {
    half* query;
    half* attn_mask;
    half* output;
    int8_t* cache;
    half* scale;
    int64_t* cachestarts;
    int64_t* kvstarts;
    float attn_scale;
    int64_t layer_idx;
    int64_t num_kv_repeats;
    int64_t page_size;
    int64_t query_stride_s;
    int64_t output_stride_s;
    int64_t mask_stride_s;
    int64_t mask_stride_h;
    int64_t cache_stride_s;
    int64_t cache_stride_l;
    int64_t cache_stride_h;
    int64_t cache_stride_kv;
    int64_t cachestarts_stride_b;

    struct {
        int32_t* block_counter;
        float* log_sum_exp;
        half* partial_out;
    } multi_block;
};


template<int32_t THREAD_GROUP_SIZE>
__device__ inline
float attn_thread_group_reduce_sum(float qk)
{
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t WPT, int32_t STOP_MASK>
__device__ inline
float attn_block_reduce_max(float reducing, float* shared_mem)
{
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= STOP_MASK; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT, int32_t STOP_MASK>
__device__ inline
float attn_block_reduce_sum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= STOP_MASK; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}


template<
    int32_t HEAD_SIZE,          // head dimension
    int32_t THREAD_GROUP_SIZE,  // how many threads inside a group (each group deal with one context)
    int32_t TPB,                // threads per block
    int32_t QUANT_GROUP,
    int32_t MULTI_BLOCK,        // do flash decoding if more than 1
    bool    ATTN_MASK,
    int32_t PAGE_SIZE>
__global__
void dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
{
    /***
    * You have to remember that this Kernel was created by a brother on the night of July 20, 2023. On that day,
    * Beijing experienced the strongest rainstorm since the beginning of summer.        --ZhiLao /doge

    DecodingAttention is a special operator designed specifically for large language models(LLM) decoding.

    It requires that the length of each input Query is always 1,
        while the Key and Value can have different lengths.

    This operator supports padding removal optimization, meaning that Q, K, and V all need to have their tokens
        concentrated in one sentence for input, with shapes like Q: [seq_lens, num_heads, head_size],
        and K: [context_lens, num_kv_heads, head_size].

    Since the Query sentence length is always 1, this operator is literally a fused matrix-vector multiplications operation.
        It does not utilize tensor cores for computation.

    The calculation logic is divided into three steps: gemv(QK) + softmax(Attention) + gemv(KV).
        In the provided code, it has already been split into these three parts.
    ***/

    /* --- Decoding Attention Kernel Implementation --- */

    // magic number for quick convert from int8 to fp16
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    static constexpr uint32_t mask_for_elt_01       = 0x5150;
    static constexpr uint32_t mask_for_elt_23       = 0x5352;
    static constexpr uint32_t start_byte_for_fp16   = 0x64646464;

    constexpr int64_t WARP_SIZE = 32;                                   // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                      // warp per thread block
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;        // thread group per warp
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT;  // thread group per thread block

    // const int64_t num_heads     = gridDim.x;
    const int64_t num_batchs    = gridDim.y;
    const int32_t head_idx      = blockIdx.x;
    const int64_t batch_idx     = blockIdx.y;
    const int64_t block_idx     = blockIdx.z;           // multi-block index for flash decoding
    constexpr int64_t VEC_SIZE  = 16 / sizeof(half);    // num of fp16 inside a 128bit vector for memory loading and storing

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE; // num of vecotr for each thread handles

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);
    static_assert(QUANT_GROUP == 8);

    constexpr int64_t QUANT_GROUP_SHIFT = 3;

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;

    const int64_t cache_offset_s  = PAGE_SIZE <= 0 ? p.cachestarts[batch_idx] : 0;  // base address of cache for each batch
    const int32_t kv_head_idx     = head_idx / int32_t(p.num_kv_repeats);           // same of head_idx if not using GQA

    fp16_t *partial_o       = nullptr;  // base address for each head to store partial output generated by flash decoding
    fp32_t *partial_log_sum = nullptr;  // partial log sum exp for flash decoding
    int32_t *block_counter  = nullptr;  // block counter for flash decoding to select the final block to do final reduction
    if (MULTI_BLOCK > 1) {
        partial_o
            = p.multi_block.partial_out
            + batch_idx * HEAD_SIZE * MULTI_BLOCK
            + head_idx * num_batchs * HEAD_SIZE * MULTI_BLOCK;
        partial_log_sum
            = p.multi_block.log_sum_exp
            + batch_idx * MULTI_BLOCK
            + head_idx * num_batchs * MULTI_BLOCK;
        block_counter
            = p.multi_block.block_counter
            + batch_idx
            + head_idx * num_batchs;
    }

    half *attn_mask = nullptr;
    if (ATTN_MASK) {
        attn_mask = p.attn_mask
                + p.mask_stride_h * head_idx
                + batch_idx * p.mask_stride_s
                + p.kvstarts[batch_idx];
    }

    // load Q from global memory to registers by vectorized accesss(128 bit)
    // every THREAD_GROUP load the same head of Q for one block
    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        copy<sizeof(half) * VEC_SIZE>(
            &p.query[
                batch_idx * p.query_stride_s +
                head_idx * HEAD_SIZE +
                (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE
            ],
            &local_q[i * VEC_SIZE]);
    }
    // ------------------------------------------------ //
    // Step 2. Solve QK Dot

    // In the process of handling the QK matrix multiplication, we will divide a complete Thread Warp into several Thread groups.
    // Each thread group reads the entire Query and saves it in registers.
    // Then, each thread group iterates through the vectors in the Key and performs dot products with the Query.
    // During this process, a WARP performs multiple vector dot product operations at once.
    // At the same time, we also record the maximum current_value of the dot product results for later use in the softmax operation.
    const int64_t context_len           = p.kvstarts[batch_idx + 1] - p.kvstarts[batch_idx];    // input context len
    const int64_t context_len_per_block = (context_len + MULTI_BLOCK - 1) / MULTI_BLOCK;        // context len for each multi-block but not the last one
    const int64_t block_context_beg     = block_idx * context_len_per_block;                    // base context index for each multi-block
    // set the valid context len for every multi-block with the last one
    const int64_t block_context_len     = context_len >= context_len_per_block * (block_idx + 1) ? context_len_per_block : context_len - block_context_beg;

    extern __shared__ float logits[];
    float partial_qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        int8_t local_k_quant[VEC_SIZE * VEC_LEN];
        half local_k_scale[VEC_LEN];
        const int64_t block_context_id = base_id + group_id;

        float qk_dot = 0.0f;

        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len) {
            const int64_t cache_token_idx = PAGE_SIZE <= 0
                            ? (cache_offset_s + block_context_beg + block_context_id)
                            : (p.cachestarts[batch_idx * p.cachestarts_stride_b + (block_context_beg + block_context_id) / PAGE_SIZE]
                                + ((block_context_beg + block_context_id) % PAGE_SIZE));
            const int64_t key_offset
                            = cache_token_idx * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                // load int8-K from kvcache to registers
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[key_idx],  &local_k_quant[i * VEC_SIZE]);
                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = p.scale[key_scale_idx];

                // fast convert from int8 to fp16
                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_k_quant[i * VEC_SIZE + k] += 128;
                }
                half result[8];
                uint32_t*      h   = reinterpret_cast<uint32_t*>(result);
                uint32_t const i8s = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_2   = reinterpret_cast<uint32_t*>(result+4);
                uint32_t const i8s_2 = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[0]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[1]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[0]) : "r"(h_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[1]) : "r"(h_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                // compute partial qk-dot in one context for each thread
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    qk_dot += __half2float(local_q[i * VEC_SIZE + j]) * __half2float(local_k_scale[i] * result[j]);
                }
            }
        }

        // every thread group get a full qk-dot in one context
        qk_dot = p.attn_scale * attn_thread_group_reduce_sum<THREAD_GROUP_SIZE>(qk_dot);

        // save qk-dot for each context and update max qk-dot
        if (group_lane_id == 0 && block_context_id < block_context_len) {
            if (ATTN_MASK)
                qk_dot += __half2float(attn_mask[block_context_beg + block_context_id]);
            logits[block_context_id] = qk_dot;
            partial_qk_max = fmaxf(qk_dot, partial_qk_max);
       }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    // The process of solving softmax is divided into two stages.
    // First, we need to reduce partial_qk_max in two dimensions: WARP and ThreadBlock.
    // Afterward, we use reduced partial_qk_max to perform softmax calculations,
    //    the results will all be stored in shared memory.
    __shared__ float red_smem[WPT];

    // reduce partial_qk_max in thread block and boardcast
    partial_qk_max = attn_block_reduce_max<WPT, 1>(partial_qk_max, red_smem);

    // Softmax Kernel Logic Start here
    // convert qk-dot to exp(local-qk-dot - max-qk-dot) in shared memory
    // sum up all exp(local-qk-dot - max-qk-dot)
    float partial_exp_sum = 0.0f;
    for (int64_t block_context_id = threadIdx.x; block_context_id < block_context_len; block_context_id += TPB){
        logits[block_context_id] -= partial_qk_max;
        logits[block_context_id] = exp(logits[block_context_id]);
        partial_exp_sum += logits[block_context_id];
    }

    // block reduce sum on partial_exp_sum
    // Warp per thread block must be power-of-2 for reducation, check attn_block_reduce_sum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    partial_exp_sum = attn_block_reduce_sum<WPT, 1>(partial_exp_sum, red_smem);

    // save partial log sum exp for flash decoding
    if (MULTI_BLOCK > 1 && threadIdx.x == 0) {
        partial_log_sum[block_idx] = partial_qk_max + log(partial_exp_sum);
    }

    // ------------------------------------------------ //
    // Step 4. Solve logits * V

    int8_t local_v_quant[VEC_SIZE * VEC_LEN];
    float local_v[VEC_SIZE * VEC_LEN];
    half local_v_scale[VEC_LEN];

    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        const int64_t block_context_id = base_id + group_id;
        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len) {
            const int64_t cache_token_idx = PAGE_SIZE <= 0
                            ? (cache_offset_s + block_context_beg + block_context_id)
                            : (p.cachestarts[batch_idx * p.cachestarts_stride_b + (block_context_beg + block_context_id) / PAGE_SIZE]
                                + ((block_context_beg + block_context_id) % PAGE_SIZE));
            const int64_t value_offset
                            = cache_token_idx * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
                            + group_lane_id * VEC_SIZE
                            + p.cache_stride_kv;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                // load int8-V from kvcache to registers
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[value_idx],  &local_v_quant[i * VEC_SIZE]);
                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = p.scale[value_scale_idx];

                // fast convert from int8 to fp16
                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_v_quant[i * VEC_SIZE + k] += 128;
                }
                half result[8];
                uint32_t*      h   = reinterpret_cast<uint32_t*>(result);
                uint32_t const i8s = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_2   = reinterpret_cast<uint32_t*>(result+4);
                uint32_t const i8s_2 = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[0]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_2[1]) : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[0]) : "r"(h_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_2[1]) : "r"(h_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                // v * sum(exp(context_qk_dot - max_qk_dot))
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_v[i * VEC_SIZE + j] += __half2float(local_v_scale[i] * result[j]) * logits[block_context_id];
                }
            }
        }
    }

    // complete softmax in local_v by dividing partial_exp_sum to generate partial output in local_v
    const float inv_sum = __fdividef(1.f, partial_exp_sum + 1e-6f);
    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] *= inv_sum;
        #pragma unroll
        for (int32_t mask = THREAD_GROUP_SIZE; mask <= WARP_SIZE >> 1; mask = mask << 1) {
            local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
        }
    }
    //for now, every warp's each thread group got the partial result inside a warp
    //we need to add up each warp's first thread group by reusing the logits smem

    // wait for logits to be reused
    __syncthreads();

    constexpr int64_t WORK_THREAD = WPT * THREAD_GROUP_SIZE * VEC_LEN;          // num of thread needed to complete block output reduction
    constexpr int64_t WORK_WARP = (WORK_THREAD + WARP_SIZE - 1) / WARP_SIZE;    // num of warp needed for reduction
    constexpr int64_t VPT = 16;                     // 16 * 8bit
    constexpr int64_t V32PT = 16 / sizeof(float);   // num of fp32 inside a vector

    const int32_t v_warp_id  = threadIdx.x % WPT;                           // warp index of reduce data for each thread to load
    const int32_t v_group_id = (threadIdx.x / WPT) % THREAD_GROUP_SIZE;     // group index of reduce data for each thread to load
    const int32_t v_vec_id   = threadIdx.x / (WPT * THREAD_GROUP_SIZE);     // vector index of reduce data for each thread to load

    half local_out[VEC_SIZE];

    // save local_v to shared memory without bank conflict
    if (warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_LEN * VEC_SIZE; i += V32PT) {
            copy<VPT>(
                &local_v[i],
                &logits[
                    i * WPT * THREAD_GROUP_SIZE +
                    warp_lane_id * WPT * V32PT +
                    ((warp_id + warp_lane_id) % WPT) * V32PT]);
        }
    }

    __syncthreads();

    // WPT reduce
    if (warp_id < WORK_WARP) {
        // each thread only load VEC_SIZE of partial ouput
        if (threadIdx.x < WORK_THREAD) {
            #pragma unroll
            for (int32_t i = 0; i < VEC_SIZE; i+= V32PT) {
                copy<VPT>(
                    &logits[
                        v_vec_id * VEC_SIZE * WPT * THREAD_GROUP_SIZE +
                        i * WPT * THREAD_GROUP_SIZE +
                        v_group_id * WPT * V32PT +
                        ((v_warp_id + v_group_id) % WPT) * V32PT],
                    &local_v[i]);
            }
        } else {
            for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i+= 1) {
                local_v[i] = 0.f;
            }
        }
        // block reduce sum on ouput
        #pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; i++) {
            #pragma unroll
            for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
                local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
            }
            local_out[i] = __float2half(local_v[i]);
        }
        if (v_warp_id == 0) {
            // save block ouput to final address or buffer for flash decoding
            half* partial_out = (MULTI_BLOCK == 1)
                    ? &p.output[
                        batch_idx * p.output_stride_s +
                        head_idx * HEAD_SIZE +
                        v_vec_id * THREAD_GROUP_SIZE * VEC_SIZE +
                        v_group_id * VEC_SIZE]
                    : &partial_o[
                        (v_vec_id * THREAD_GROUP_SIZE + v_group_id) * MULTI_BLOCK * VEC_SIZE
                        + block_idx * VEC_SIZE];
            copy<VPT>(local_out, partial_out);
        }
    }

    // Flash decoding
    if (MULTI_BLOCK > 1) {
        __syncthreads();

        bool last_block = false;
        // Make sure every block finishs the partial computation.
        if (threadIdx.x == 0) {
            if (atomicAdd(block_counter, 1) == MULTI_BLOCK - 1) {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx = threadIdx.x % MULTI_BLOCK;

            // get max block log sum exp
            float local_log_sum_exp = warp_lane_id < MULTI_BLOCK ? partial_log_sum[multi_block_idx] : -FLT_MAX;
            float max_log_sum_exp = local_log_sum_exp;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                max_log_sum_exp = fmaxf(max_log_sum_exp, __shfl_xor_sync(uint32_t(-1), max_log_sum_exp, mask));
            }
            max_log_sum_exp = __shfl_sync(uint32_t(-1), max_log_sum_exp, 0);

            // update scale
            float local_scale = warp_lane_id < MULTI_BLOCK ? exp(local_log_sum_exp - max_log_sum_exp) : 0.f;
            float scale_sum = local_scale;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                scale_sum += __shfl_xor_sync(uint32_t(-1), scale_sum, mask);
            }
            scale_sum = __shfl_sync(uint32_t(-1), scale_sum, 0);

            float *scale_smem = logits;
            int scale_id = warp_id * MULTI_BLOCK + warp_lane_id;
            if (warp_lane_id < MULTI_BLOCK && scale_id < WARP_SIZE) {
                scale_smem[scale_id] = local_scale / scale_sum;
            }
            __syncthreads();

            // final reduce for multi-block output
            const int64_t head_dim_idx_base = threadIdx.x / MULTI_BLOCK * VEC_SIZE;
            const int64_t head_dim_idx_stride = TPB / MULTI_BLOCK * VEC_SIZE;

            for (int64_t head_dim_idx = head_dim_idx_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_idx_stride) {
                half final_out[VEC_SIZE];
                local_scale = scale_smem[warp_lane_id];
                copy<VEC_SIZE*sizeof(half)>(
                    &partial_o[
                        head_dim_idx * MULTI_BLOCK +
                        multi_block_idx * VEC_SIZE],
                    final_out);

                #pragma unroll
                for (int32_t i = 0; i < VEC_SIZE; i++) {
                    float float_out = __half2float(final_out[i]) * local_scale;
                    # pragma unroll
                    for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                        float_out += __shfl_xor_sync(uint32_t(-1), float_out, mask);
                    }
                    final_out[i] = __float2half(float_out);
                }

                if (multi_block_idx == 0) {
                    copy<VPT>(
                        final_out,
                        &p.output[
                            batch_idx * p.output_stride_s +
                            head_idx * HEAD_SIZE +
                            head_dim_idx]);
                }
            }
        }
    }
}


template<
    int32_t HEAD_SIZE,          // head dimension
    int32_t THREAD_GROUP_SIZE,  // how many threads inside a group
    int32_t TPB,                // threads per block
    int32_t QUANT_GROUP,
    int32_t MULTI_BLOCK,        // do flash decoding if more than 1
    bool    ATTN_MASK,
    int32_t PAGE_SIZE>
__global__
void dynamic_batching_decoding_cache_infinity_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
{
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    static constexpr uint32_t mask_for_elt_01       = 0x5150;
    static constexpr uint32_t mask_for_elt_23       = 0x5352;
    static constexpr uint32_t start_byte_for_fp16   = 0x64646464;

    constexpr int64_t WARP_SIZE = 32;                              // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                 // warp per thread block
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;       // thread group per warp
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT; // thread group per thread block

    // const int64_t num_heads     = gridDim.x;
    const int64_t num_batchs    = gridDim.y;
    const int32_t head_idx      = blockIdx.x;
    const int64_t batch_idx     = blockIdx.y;
    const int64_t block_idx     = blockIdx.z;
    constexpr int64_t VEC_SIZE  = 16 / sizeof(half);  // 128 bits

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE;

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);
    static_assert(QUANT_GROUP == 8);

    constexpr int64_t QUANT_GROUP_SHIFT = 3;

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    half local_q[VEC_SIZE * VEC_LEN];

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;

    const int64_t cache_offset_s  = PAGE_SIZE <= 0 ? p.cachestarts[batch_idx] : 0;
    const int32_t kv_head_idx     = head_idx / p.num_kv_repeats;

    fp16_t *partial_o       = nullptr;
    fp32_t *partial_log_sum = nullptr;
    int32_t *block_counter  = nullptr;
    if (MULTI_BLOCK > 1) {
        partial_o
            = p.multi_block.partial_out
            + batch_idx * HEAD_SIZE * MULTI_BLOCK
            + head_idx * num_batchs * HEAD_SIZE * MULTI_BLOCK;
        partial_log_sum
            = p.multi_block.log_sum_exp
            + batch_idx * MULTI_BLOCK
            + head_idx * num_batchs * MULTI_BLOCK;
        block_counter
            = p.multi_block.block_counter
            + batch_idx
            + head_idx * num_batchs;
    }

    half *attn_mask = nullptr;
    if (ATTN_MASK) {
        attn_mask = p.attn_mask
                + p.mask_stride_h * head_idx
                + batch_idx * p.mask_stride_s
                + p.kvstarts[batch_idx];
    }

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        copy<sizeof(half) * VEC_SIZE>(
            &p.query[
                batch_idx * p.query_stride_s +
                head_idx * HEAD_SIZE +
                (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE
            ],
            &local_q[i * VEC_SIZE]);
    }

    const int64_t context_len           = p.kvstarts[batch_idx + 1] - p.kvstarts[batch_idx];
    const int64_t context_len_per_block = (context_len + MULTI_BLOCK - 1) / MULTI_BLOCK;
    const int64_t block_context_beg     = block_idx * context_len_per_block;
    const int64_t block_context_len     = context_len >= context_len_per_block * (block_idx + 1) ? context_len_per_block : context_len - block_context_beg;

    __shared__ float tmp_buffer[WPT * HEAD_SIZE];
    float thread_qk_max = -FLT_MAX;
    float partial_exp_sum = 0.0f;

    float local_v[VEC_SIZE * VEC_LEN];
    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < block_context_len; base_id += GPT) {
        float local_v_new[VEC_SIZE * VEC_LEN];
        int8_t local_k_quant[VEC_SIZE * VEC_LEN], local_v_quant[VEC_SIZE * VEC_LEN];
        half local_k_scale[VEC_LEN], local_v_scale[VEC_LEN];
        const int64_t block_context_id = base_id + group_id;

        float qk_dot = 0.0f;

        // all thread groups within a warp must be launched together.
        if (block_context_id < block_context_len) {
            const int64_t cache_token_idx = PAGE_SIZE <= 0
                            ? (cache_offset_s + block_context_beg + block_context_id)
                            : (p.cachestarts[batch_idx * p.cachestarts_stride_b + (block_context_beg + block_context_id) / PAGE_SIZE]
                                + ((block_context_beg + block_context_id) % PAGE_SIZE));
            const int64_t key_offset
                            = cache_token_idx * p.cache_stride_s
                            + p.layer_idx * p.cache_stride_l
                            + p.cache_stride_h * kv_head_idx
                            + group_lane_id * VEC_SIZE;
            const int64_t value_offset = key_offset + p.cache_stride_kv;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[key_idx],  &local_k_quant[i * VEC_SIZE]);
                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = p.scale[key_scale_idx];

                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&p.cache[value_idx],  &local_v_quant[i * VEC_SIZE]);
                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = p.scale[value_scale_idx];

                #pragma unroll
                for(int64_t k = 0; k < VEC_SIZE; k++) {
                    local_k_quant[i * VEC_SIZE + k] += 128;
                    local_v_quant[i * VEC_SIZE + k] += 128;
                }

                half result_k[8];
                uint32_t*      h_k   = reinterpret_cast<uint32_t*>(result_k);
                uint32_t const i8s_k = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[0]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[1]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[0]) : "r"(h_k[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[1]) : "r"(h_k[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_k_2   = reinterpret_cast<uint32_t*>(result_k+4);
                uint32_t const i8s_k_2 = reinterpret_cast<uint32_t const&>(*(local_k_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k_2[0]) : "r"(i8s_k_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k_2[1]) : "r"(i8s_k_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k_2[0]) : "r"(h_k_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k_2[1]) : "r"(h_k_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    qk_dot += __half2float(local_q[i * VEC_SIZE + j]) * __half2float(local_k_scale[i] * result_k[j]);
                }

                half result_v[8];
                uint32_t*      h_v   = reinterpret_cast<uint32_t*>(result_v);
                uint32_t const i8s_v = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[0]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[1]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[0]) : "r"(h_v[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[1]) : "r"(h_v[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                uint32_t*      h_v_2   = reinterpret_cast<uint32_t*>(result_v+4);
                uint32_t const i8s_v_2 = reinterpret_cast<uint32_t const&>(*(local_v_quant + i * VEC_SIZE + 4));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v_2[0]) : "r"(i8s_v_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v_2[1]) : "r"(i8s_v_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v_2[0]) : "r"(h_v_2[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v_2[1]) : "r"(h_v_2[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_v_new[i * VEC_SIZE + j] = __half2float(local_v_scale[i] * result_v[j]);
                }
            }
        }

        qk_dot = p.attn_scale * attn_thread_group_reduce_sum<THREAD_GROUP_SIZE>(qk_dot);

        if (block_context_id < block_context_len) {
            if (ATTN_MASK) {
                qk_dot += __half2float(attn_mask[block_context_beg + block_context_id]);
            }
            // Computing inside performs better since using one fma per iteration
            if (qk_dot > thread_qk_max) {
                float logit_scale = exp(thread_qk_max - qk_dot);
                thread_qk_max = qk_dot;
                partial_exp_sum = partial_exp_sum * logit_scale + 1.f;
                #pragma unroll
                for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
                    local_v[i] = local_v[i] * logit_scale + local_v_new[i];
                }
            } else {
                float logit_scale = exp(qk_dot - thread_qk_max);
                partial_exp_sum += logit_scale;
                #pragma unroll
                for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
                    local_v[i] = local_v[i] + local_v_new[i] * logit_scale;
                }
            }
        }
    }

    // reduce partial_qk_max in thread block and boardcast
    float partial_qk_max = attn_block_reduce_max<WPT, THREAD_GROUP_SIZE>(thread_qk_max, tmp_buffer);

    if (partial_qk_max > thread_qk_max) {
        float logit_scale = exp(thread_qk_max - partial_qk_max);
        partial_exp_sum *= logit_scale;
        #pragma unroll
        for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
            local_v[i] *= logit_scale;
        }
    }

    // block reduce sum on partial_exp_sum
    // Warp per thread block must be power-of-2 for reducation, check attn_block_reduce_sum kernel.
    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    partial_exp_sum = attn_block_reduce_sum<WPT, THREAD_GROUP_SIZE>(partial_exp_sum, &tmp_buffer[WPT]);

    if (MULTI_BLOCK > 1 && threadIdx.x == 0) {
        partial_log_sum[block_idx] = partial_qk_max + log(partial_exp_sum);
    }

    const float inv_sum = __fdividef(1.f, partial_exp_sum + 1e-6f);
    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] *= inv_sum;
        #pragma unroll
        for (int32_t mask = THREAD_GROUP_SIZE; mask <= WARP_SIZE >> 1; mask = mask << 1) {
            local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
        }
    }

    // wait for logits to be reused
    __syncthreads();

    constexpr int64_t WORK_WARP = (WPT * THREAD_GROUP_SIZE * VEC_LEN + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int64_t VPT   = 16;
    constexpr int64_t V32PT = 16 / sizeof(float);

    const int32_t v_warp_id  = threadIdx.x % WPT;
    const int32_t v_group_id = (threadIdx.x / WPT) % THREAD_GROUP_SIZE;
    const int32_t v_vec_id   = threadIdx.x / (WPT * THREAD_GROUP_SIZE);

    half local_out[VEC_SIZE];

    // save local_v to shared memory
    if (warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_LEN * VEC_SIZE; i += V32PT) {
            copy<VPT>(
                &local_v[i],
                &tmp_buffer[
                    i * WPT * THREAD_GROUP_SIZE +
                    warp_lane_id * WPT * V32PT +
                    ((warp_id + warp_lane_id) % WPT) * V32PT]);
        }
    }

    __syncthreads();

    // WPT reduce
    if (warp_id < WORK_WARP) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; i+= V32PT) {
            copy<VPT>(
                &tmp_buffer[
                    v_vec_id * VEC_SIZE * WPT * THREAD_GROUP_SIZE +
                    i * WPT * THREAD_GROUP_SIZE +
                    v_group_id * WPT * V32PT +
                    ((v_warp_id + v_group_id) % WPT) * V32PT],
                &local_v[i]);
        }
        #pragma unroll
        for (int32_t i = 0; i < VEC_SIZE; i++) {
            #pragma unroll
            for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
                local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
            }
            local_out[i] = __float2half(local_v[i]);
        }
        if (v_warp_id == 0) {
            half* partial_out = (MULTI_BLOCK == 1)
                    ? &p.output[
                        batch_idx * p.output_stride_s +
                        head_idx * HEAD_SIZE +
                        v_vec_id * THREAD_GROUP_SIZE * VEC_SIZE +
                        v_group_id * VEC_SIZE]
                    : &partial_o[
                        (v_vec_id * THREAD_GROUP_SIZE + v_group_id) * MULTI_BLOCK * VEC_SIZE
                        + block_idx * VEC_SIZE];
            copy<VPT>(local_out, partial_out);
        }
    }

    // Flash decoding
    if (MULTI_BLOCK > 1) {
        __syncthreads();

        bool last_block = false;
        // Make sure every block finishs the partial computation.
        if (threadIdx.x == 0) {
            if (atomicAdd(block_counter, 1) == MULTI_BLOCK - 1) {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx = threadIdx.x % MULTI_BLOCK;

            float local_log_sum_exp = warp_lane_id < MULTI_BLOCK ? partial_log_sum[multi_block_idx] : -FLT_MAX;
            float max_log_sum_exp = local_log_sum_exp;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                max_log_sum_exp = fmaxf(max_log_sum_exp, __shfl_xor_sync(uint32_t(-1), max_log_sum_exp, mask));
            }
            max_log_sum_exp = __shfl_sync(uint32_t(-1), max_log_sum_exp, 0);

            float local_scale = warp_lane_id < MULTI_BLOCK ? exp(local_log_sum_exp - max_log_sum_exp) : 0.f;
            float scale_sum = local_scale;
            # pragma unroll
            for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                scale_sum += __shfl_xor_sync(uint32_t(-1), scale_sum, mask);
            }
            scale_sum = __shfl_sync(uint32_t(-1), scale_sum, 0);

            int scale_id = warp_id * MULTI_BLOCK + warp_lane_id;
            if (warp_lane_id < MULTI_BLOCK && scale_id < WARP_SIZE) {
                tmp_buffer[scale_id] = local_scale / scale_sum;
            }
            __syncthreads();

            const int64_t head_dim_idx_base = threadIdx.x / MULTI_BLOCK * VEC_SIZE;
            const int64_t head_dim_idx_stride = TPB / MULTI_BLOCK * VEC_SIZE;

            for (int64_t head_dim_idx = head_dim_idx_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_idx_stride) {
                half final_out[VEC_SIZE];
                local_scale = tmp_buffer[warp_lane_id];
                copy<VEC_SIZE*sizeof(half)>(
                    &partial_o[
                        head_dim_idx * MULTI_BLOCK +
                        multi_block_idx * VEC_SIZE],
                    final_out);

                #pragma unroll
                for (int32_t i = 0; i < VEC_SIZE; i++) {
                    float float_out = __half2float(final_out[i]) * local_scale;
                    # pragma unroll
                    for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                        float_out += __shfl_xor_sync(uint32_t(-1), float_out, mask);
                    }
                    final_out[i] = __float2half(float_out);
                }

                if (multi_block_idx == 0) {
                    copy<VPT>(
                        final_out,
                        &p.output[
                            batch_idx * p.output_stride_s +
                            head_idx * HEAD_SIZE +
                            head_dim_idx]);
                }
            }
        }
    }
}

template<
    int32_t HEAD_SIZE,
    int32_t TPB,
    int32_t FULL_GROUP_SIZE,
    int32_t TAIL_GROUP_SIZE,
    bool    IS_TAIL_GROUP
>
__device__ inline
void attn_load_group_query(fp16_t* q_loc, fp16_t* q_glb, fp16_t* q_shm)
{
    constexpr int64_t WARP_SIZE = 32;
    constexpr int64_t MMA_TPG   = 4;
    constexpr int64_t F16PV     = 16 / sizeof(fp16_t);
    constexpr int64_t Q_SIZE    = FULL_GROUP_SIZE * HEAD_SIZE / WARP_SIZE;

    constexpr int64_t VALID_GROUP_SIZE = IS_TAIL_GROUP ? TAIL_GROUP_SIZE : FULL_GROUP_SIZE;

    const int64_t tid           = threadIdx.x;
    const int64_t warp_lane_id  = tid % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / MMA_TPG / 2 + warp_lane_id / MMA_TPG % 2 * 4;
    // const int64_t group_id      = warp_lane_id / MMA_TPG;
    const int64_t group_lane_id = warp_lane_id % MMA_TPG;

    #pragma unroll
    for (int64_t i = 0; i < HEAD_SIZE * VALID_GROUP_SIZE / (TPB * F16PV); i++) {
        int64_t query_group_id  = (tid + i) * F16PV / HEAD_SIZE;
        int64_t head_dim_id     = (tid + i) * F16PV % HEAD_SIZE;
        copy<sizeof(fp16_t) * F16PV>(
            &q_glb[(tid + i) * F16PV],
            &q_shm[query_group_id * (HEAD_SIZE + F16PV) + head_dim_id]);
    }
    if (HEAD_SIZE * VALID_GROUP_SIZE % (TPB * F16PV)) {
        int64_t query_offset
                    = tid * F16PV
                    + HEAD_SIZE * VALID_GROUP_SIZE
                    - HEAD_SIZE * VALID_GROUP_SIZE % (TPB * F16PV);
        if (query_offset < HEAD_SIZE * VALID_GROUP_SIZE) {
            int64_t query_group_id  = query_offset / HEAD_SIZE;
            int64_t head_dim_id     = query_offset % HEAD_SIZE;
            copy<sizeof(fp16_t) * F16PV>(
                &q_glb[query_offset],
                &q_shm[query_group_id * (HEAD_SIZE + F16PV) + head_dim_id]);
        }
    }

    __syncthreads();

    if (IS_TAIL_GROUP) {
        if (group_id < TAIL_GROUP_SIZE) {
            #pragma unroll
            for (int64_t i = 0; i < Q_SIZE; i += F16PV) {
                copy<sizeof(fp16_t) * F16PV>(
                    &q_shm[group_id * (HEAD_SIZE + F16PV) + group_lane_id * Q_SIZE + i],
                    &q_loc[i]);
            }
        } else {
            uint32_t* h_q = reinterpret_cast<uint32_t*>(q_loc);
            #pragma unroll
            for (int64_t i = 0; i < Q_SIZE / 2; i++) {
                h_q[i] = 0;
            }
        }
    } else {
        #pragma unroll
        for (int64_t i = 0; i < Q_SIZE; i += F16PV) {
            copy<sizeof(fp16_t) * F16PV>(
                &q_shm[group_id * (HEAD_SIZE + F16PV) + group_lane_id * Q_SIZE + i],
                &q_loc[i]);
        }
    }
}

template<int32_t LOGITS_SIZE>
__device__ inline
void attn_logits_reorder(fp16_t* dst, fp32_t* src, const int32_t idx1, const int32_t idx2)
{
    constexpr int32_t WARP_SIZE         = 32;
    constexpr int32_t HALF_WARP_SIZE    = WARP_SIZE / 2;
    constexpr int32_t HALF_LOGITS_SIZE  = LOGITS_SIZE / 2;

    const int32_t lane_id = threadIdx.x % WARP_SIZE;

    fp16_t tmp[HALF_LOGITS_SIZE];
    uint32_t* ht = reinterpret_cast<uint32_t*>(tmp);
    uint32_t* hd = reinterpret_cast<uint32_t*>(dst);

    // step 1
    if (lane_id < HALF_WARP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < HALF_LOGITS_SIZE; i++) {
            tmp[i] = __float2half(src[2 * i + 1]);
            dst[2 * i] = __float2half(src[2 * i]);
        }
    } else {
        #pragma unroll
        for (int32_t i = 0; i < HALF_LOGITS_SIZE; i++) {
            tmp[i] = __float2half(src[2 * i]);
            dst[2 * i + 1] = __float2half(src[2 * i + 1]);
        }
    }
    #pragma unroll
    for (int32_t i = 0; i < HALF_LOGITS_SIZE / 2; i++) {
        ht[i] = __shfl_sync(uint32_t(-1), ht[i], idx1);
    }

    // step 2
    if (lane_id < HALF_WARP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < HALF_LOGITS_SIZE; i++) {
            dst[2 * i + 1] = tmp[i];
        }
    } else {
        #pragma unroll
        for (int32_t i = 0; i < HALF_LOGITS_SIZE; i++) {
            dst[2 * i] = tmp[i];
        }
    }
    #pragma unroll
    for (int32_t i = 0; i < HALF_LOGITS_SIZE; i++) {
        hd[i] = __shfl_sync(uint32_t(-1), hd[i], idx2);
    }
}

template<int32_t WPT, int32_t GROUP_SIZE, int32_t VEC_SIZE>
__device__ inline
void attn_block_reduce_group_max(fp32_t *dst, fp32_t *src, fp32_t *shared_mem)
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t LOAD_SIZE = WPT * VEC_SIZE * GROUP_SIZE / WARP_SIZE;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id < GROUP_SIZE) {
        copy<sizeof(fp32_t) * VEC_SIZE>(src, &shared_mem[warp_id * GROUP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }
    __syncthreads();

    fp32_t tmp[LOAD_SIZE];
    #pragma unroll
    for (int32_t i = 0; i < LOAD_SIZE; i += VEC_SIZE) {
        copy<sizeof(fp32_t) * VEC_SIZE>(&shared_mem[i * WARP_SIZE + lane_id * VEC_SIZE], &tmp[i]);
    }

    #pragma unroll
    for (int32_t i = 1; i < LOAD_SIZE / VEC_SIZE; i++) {
        #pragma unroll
        for (int32_t j = 0; j < VEC_SIZE; j++) {
            tmp[j] = fmaxf(tmp[j], tmp[i * VEC_SIZE + j]);
        }
    }

    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE; i++) {
        dst[i] = tmp[i];
        #pragma unroll
        for (int32_t mask = GROUP_SIZE; mask < WARP_SIZE; mask <<= 1) {
            dst[i] = fmaxf(dst[i], __shfl_xor_sync(uint32_t(-1), dst[i], mask));
        }
    }
}

template<int32_t WPT, int32_t GROUP_SIZE, int32_t VEC_SIZE>
__device__ inline
void attn_block_reduce_group_sum(fp32_t *reducing, fp32_t *shared_mem)
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t LOAD_SIZE = WPT * VEC_SIZE * GROUP_SIZE / WARP_SIZE;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE; i++) {
        #pragma unroll
        for (int32_t mask = GROUP_SIZE; mask < WARP_SIZE; mask <<= 1) {
            reducing[i] += __shfl_xor_sync(uint32_t(-1), reducing[i], mask);
        }
    }

    if (lane_id < GROUP_SIZE) {
        copy<sizeof(fp32_t) * VEC_SIZE>(reducing, &shared_mem[warp_id * GROUP_SIZE * VEC_SIZE + lane_id * VEC_SIZE]);
    }
    __syncthreads();

    fp32_t tmp[LOAD_SIZE];
    #pragma unroll
    for (int32_t i = 0; i < LOAD_SIZE; i += VEC_SIZE) {
        copy<sizeof(fp32_t) * VEC_SIZE>(&shared_mem[i * WARP_SIZE + lane_id * VEC_SIZE], &tmp[i]);
    }

    #pragma unroll
    for (int32_t i = 1; i < LOAD_SIZE / VEC_SIZE; i++) {
        #pragma unroll
        for (int32_t j = 0; j < VEC_SIZE; j++) {
            tmp[j] += tmp[i * VEC_SIZE + j];
        }
    }
    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE; i++) {
        reducing[i] = tmp[i];
        #pragma unroll
        for (int32_t mask = GROUP_SIZE; mask < WARP_SIZE; mask <<= 1) {
            reducing[i] += __shfl_xor_sync(uint32_t(-1), reducing[i], mask);
        }
    }
}


template<
    int32_t HEAD_SIZE,
    int32_t TPB,
    int32_t QUANT_GROUP,
    int32_t QUERY_GROUP,
    int32_t MULTI_BLOCK,    // do flash decoding if more than 1
    bool    ATTN_MASK,
    int32_t PAGE_SIZE>
__global__
void dynamic_batching_decoding_group_query_cache_attention_fp16_kernel(dynamic_batching_decoding_cache_attention_kernel_param p)
{
    static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
    static constexpr uint32_t mask_for_elt_01       = 0x5150;
    static constexpr uint32_t mask_for_elt_23       = 0x5352;
    static constexpr uint32_t start_byte_for_fp16   = 0x64646464;

    constexpr int64_t WARP_SIZE = 32;                   // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;      // warp per thread block

    constexpr int64_t VEC_WIDTH = 16;                           // 128 bits
    constexpr int64_t I8PV      = VEC_WIDTH / sizeof(int8_t);   // num per vector for int_8
    constexpr int64_t F16PV     = VEC_WIDTH / sizeof(fp16_t);   // num per vector for fp16_t
    constexpr int64_t F32PV     = VEC_WIDTH / sizeof(fp32_t);   // num per vector for fp32_t

    // HMMA.16816.F32
    constexpr int64_t MMA_M     = 16;
    constexpr int64_t MMA_N     = 8;
    constexpr int64_t MMA_K     = 16;

    constexpr int64_t MMA_TPG   = 4;                    // every 4 threads handle one major line(K) in MMA
    constexpr int64_t MMA_GPT   = 2;                    // each thread handles 2 query group
    constexpr int64_t MMA_GPW   = WARP_SIZE / MMA_TPG;  // thread group per warp

    constexpr int64_t CONTEXT_STRIDE    = MMA_M;
    constexpr int64_t FULL_GROUP_SIZE   = MMA_N;
    constexpr int64_t TAIL_GROUP_SIZE   = QUERY_GROUP % FULL_GROUP_SIZE;
    constexpr int64_t GROUP_BLOCK_SIZE  = (QUERY_GROUP + FULL_GROUP_SIZE - 1) / FULL_GROUP_SIZE;

    constexpr int64_t MMA_QPT   = MMA_N * MMA_K / WARP_SIZE;
    constexpr int64_t MMA_KPT   = MMA_M * MMA_K / WARP_SIZE;
    constexpr int64_t MMA_LPT   = MMA_M * MMA_N / WARP_SIZE;
    constexpr int64_t MMA_VPT   = MMA_M * MMA_K / WARP_SIZE;
    constexpr int64_t MMA_OPT   = MMA_M * MMA_N / WARP_SIZE;
    constexpr int64_t MMA_LPG   = MMA_LPT / MMA_GPT;
    constexpr int64_t MMA_VPG   = MMA_VPT / MMA_GPT;
    constexpr int64_t MMA_OPG   = MMA_OPT / MMA_GPT;

    constexpr int64_t Q_SIZE    = MMA_N * HEAD_SIZE / WARP_SIZE;
    constexpr int64_t K_SIZE    = MMA_M * HEAD_SIZE / WARP_SIZE;
    constexpr int64_t L_SIZE    = MMA_M * MMA_N / WARP_SIZE;
    constexpr int64_t V_SIZE    = MMA_M * HEAD_SIZE / WARP_SIZE;
    constexpr int64_t O_SIZE    = MMA_N * HEAD_SIZE / WARP_SIZE;

    constexpr int64_t K_PER_CONTEXT     = HEAD_SIZE / MMA_TPG;
    constexpr int64_t V_PER_CONTEXT     = HEAD_SIZE / MMA_GPW;
    constexpr int64_t O_PER_QUERY       = HEAD_SIZE / MMA_GPW;
    constexpr int64_t KS_PER_CONTEXT    = K_PER_CONTEXT / QUANT_GROUP;
    constexpr int64_t VS_PER_CONTEXT    = (V_PER_CONTEXT + QUANT_GROUP - 1) / QUANT_GROUP;

    constexpr int64_t K_LOAD_STRIDE
                        = (K_PER_CONTEXT % I8PV == 0) ? I8PV
                        : (K_PER_CONTEXT % (I8PV / 2) == 0) ? (I8PV / 2)
                        : (I8PV / 4);
    constexpr int64_t V_LOAD_STRIDE
                        = (V_PER_CONTEXT % I8PV == 0) ? I8PV
                        : (V_PER_CONTEXT % (I8PV / 2) == 0) ? (I8PV / 2)
                        : (I8PV / 4);
    constexpr int64_t KS_LOAD_STRIDE
                        = (KS_PER_CONTEXT % F16PV == 0) ? F16PV
                        : (KS_PER_CONTEXT % (F16PV / 2) == 0) ? (F16PV / 2)
                        : (KS_PER_CONTEXT % (F16PV / 4) == 0) ? (F16PV / 4)
                        : 1;
    constexpr int64_t VS_LOAD_STRIDE
                        = (V_PER_CONTEXT % QUANT_GROUP) ? 1
                        : (VS_PER_CONTEXT % F16PV == 0) ? F16PV
                        : (VS_PER_CONTEXT % (F16PV / 2) == 0) ? (F16PV / 2)
                        : (VS_PER_CONTEXT % (F16PV / 4) == 0) ? (F16PV / 4)
                        : 1;

    constexpr int64_t VALID_LOGIT_SIZE  = (QUERY_GROUP <= FULL_GROUP_SIZE / MMA_GPT) ? 1 : 2;
    constexpr int64_t VALID_REDUCE_HEAD = (QUERY_GROUP <= FULL_GROUP_SIZE) ? QUERY_GROUP : FULL_GROUP_SIZE;
    constexpr int64_t HEAD_PER_REDUCE
                        = (WPT * HEAD_SIZE > 4096) ? 1
                        : (WPT * HEAD_SIZE > 2048) ? 2
                        : (WPT * HEAD_SIZE > 1024) ? 4
                        : 8;
    constexpr int64_t THREAD_PER_REDUCE = WPT * HEAD_PER_REDUCE * HEAD_SIZE / F16PV;
    constexpr int64_t THREAD_PER_HEAD   = HEAD_SIZE / F16PV;

    const int64_t num_batchs    = gridDim.y;
    const int64_t batch_idx     = blockIdx.y;
    const int64_t block_idx     = blockIdx.z;
    const int32_t qo_head_base  = (QUERY_GROUP <= FULL_GROUP_SIZE)
                    ? blockIdx.x * QUERY_GROUP
                    : blockIdx.x * FULL_GROUP_SIZE - blockIdx.x / GROUP_BLOCK_SIZE * (FULL_GROUP_SIZE - TAIL_GROUP_SIZE);
    const int32_t kv_head_idx   = (QUERY_GROUP <= FULL_GROUP_SIZE)
                    ? blockIdx.x
                    : blockIdx.x / GROUP_BLOCK_SIZE;

    const int64_t tid           = threadIdx.x;
    const int64_t warp_id       = tid / WARP_SIZE;
    const int64_t warp_lane_id  = tid % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / MMA_TPG;
    const int64_t group_lane_id = warp_lane_id % MMA_TPG;

    const bool is_tail_group = (QUERY_GROUP < FULL_GROUP_SIZE)
                    ? true
                    : ((TAIL_GROUP_SIZE > 0) && (blockIdx.x % GROUP_BLOCK_SIZE == GROUP_BLOCK_SIZE - 1));

    const int64_t logit_reorder_id1 = (warp_lane_id + WARP_SIZE / 2) % WARP_SIZE;
    const int64_t logit_reorder_id2
                    = warp_lane_id / MMA_TPG % 2 * 16
                    + warp_lane_id / MMA_TPG / 2
                    + warp_lane_id % MMA_TPG * 4;

    const int64_t context_len           = p.kvstarts[batch_idx + 1] - p.kvstarts[batch_idx];
    const int64_t context_len_per_block = (context_len + MULTI_BLOCK - 1) / MULTI_BLOCK;
    const int64_t block_context_beg     = block_idx * context_len_per_block;
    const int64_t block_context_len     = (context_len >= context_len_per_block * (block_idx + 1))
                    ? context_len_per_block
                    : context_len - block_context_beg;

    const int64_t cache_token_base = (PAGE_SIZE <= 0)
                    ? p.cachestarts[batch_idx] + block_context_beg
                    : 0;
    const int64_t cahce_offset
                    = cache_token_base * p.cache_stride_s
                    + p.layer_idx * p.cache_stride_l
                    + kv_head_idx * p.cache_stride_h;

    fp16_t *q_glb       = p.query + batch_idx * p.query_stride_s + qo_head_base * HEAD_SIZE;
    int8_t *k_glb       = p.cache + cahce_offset + group_lane_id * K_PER_CONTEXT;
    int8_t *v_glb       = p.cache + cahce_offset + group_id * V_PER_CONTEXT + p.cache_stride_kv;
    fp16_t *k_scale_glb = p.scale + (cahce_offset + group_lane_id * K_PER_CONTEXT) / QUANT_GROUP;
    fp16_t *v_scale_glb = p.scale + (cahce_offset + group_id * V_PER_CONTEXT + p.cache_stride_kv) / QUANT_GROUP;
    fp16_t *o_glb       = p.output + batch_idx * p.output_stride_s + qo_head_base * HEAD_SIZE;

    fp16_t *partial_o       = nullptr;
    fp32_t *partial_log_sum = nullptr;
    int32_t *block_counter  = nullptr;
    if (MULTI_BLOCK > 1) {
        partial_o
            = p.multi_block.partial_out
            + batch_idx * HEAD_SIZE * MULTI_BLOCK
            + qo_head_base * num_batchs * HEAD_SIZE * MULTI_BLOCK;
        partial_log_sum
            = p.multi_block.log_sum_exp
            + batch_idx * MULTI_BLOCK
            + qo_head_base * num_batchs * MULTI_BLOCK;
        block_counter
            = p.multi_block.block_counter
            + batch_idx
            + qo_head_base * num_batchs;
    }

    fp16_t *attn_mask = nullptr;
    if (ATTN_MASK) {
        attn_mask
            = p.attn_mask
            + qo_head_base * p.mask_stride_h
            + batch_idx * p.mask_stride_s
            + p.kvstarts[batch_idx]
            + block_context_beg;
    }

    __shared__ fp32_t tmp_buffer[HEAD_PER_REDUCE * (WPT * HEAD_SIZE + F32PV)];
    fp16_t *q_shm = reinterpret_cast<fp16_t*>(tmp_buffer);
    fp16_t q_loc[Q_SIZE];
    fp32_t warp_qk_max[MMA_LPG], warp_exp_sum[MMA_LPG], warp_o[O_SIZE];

    if (is_tail_group) {
        attn_load_group_query<HEAD_SIZE, TPB, FULL_GROUP_SIZE, TAIL_GROUP_SIZE, true>(q_loc, q_glb, q_shm);
    } else {
        attn_load_group_query<HEAD_SIZE, TPB, FULL_GROUP_SIZE, TAIL_GROUP_SIZE, false>(q_loc, q_glb, q_shm);
    }

    #pragma unroll
    for (int64_t i = 0; i < MMA_LPG; i++) {
        warp_qk_max[i]  = -FLT_MAX;
        warp_exp_sum[i] = 0.f;
    }
    #pragma unroll
    for (int64_t i = 0; i < O_SIZE; i++) {
        warp_o[i] = 0.f;
    }

    for (int64_t base_id = warp_id * CONTEXT_STRIDE; base_id < block_context_len; base_id += WPT * CONTEXT_STRIDE) {
        fp32_t qk_dot[L_SIZE], tile_qk_max[MMA_LPG];
        fp16_t k_loc_reordered[K_SIZE], v_loc_reordered[V_SIZE];

        #pragma unroll
        for (int64_t i = 0; i < L_SIZE; i++) {
            qk_dot[i] = 0.f;
        }

        #pragma unroll
        for (int64_t k_context_group = 0; k_context_group < MMA_GPT; k_context_group++) {
            const int64_t k_context_id = base_id + group_id + k_context_group * MMA_GPW;
            if (k_context_id < block_context_len) {
                int8_t k_quant[K_PER_CONTEXT];
                fp16_t k_scale_loc[KS_PER_CONTEXT], k_loc[K_PER_CONTEXT];
                const int64_t cache_token_idx = (PAGE_SIZE <= 0)
                                ? k_context_id
                                : (p.cachestarts[batch_idx * p.cachestarts_stride_b + (block_context_beg + k_context_id) / PAGE_SIZE]
                                    + ((block_context_beg + k_context_id) % PAGE_SIZE));
                const int64_t key_offset = cache_token_idx * p.cache_stride_s;

                #pragma unroll
                for (int64_t i = 0; i < KS_PER_CONTEXT; i += KS_LOAD_STRIDE) {
                    copy<sizeof(fp16_t) * KS_LOAD_STRIDE>(&k_scale_glb[key_offset / QUANT_GROUP + i],  &k_scale_loc[i]);
                }

                #pragma unroll
                for (int64_t i = 0; i < K_PER_CONTEXT; i += K_LOAD_STRIDE) {
                    const int64_t key_idx = key_offset + i;
                    copy<sizeof(int8_t) * K_LOAD_STRIDE>(&k_glb[key_idx],  &k_quant[i]);

                    #pragma unroll
                    for (int64_t j = 0; j < K_LOAD_STRIDE; j++) {
                        k_quant[i + j] += 128;
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < K_LOAD_STRIDE; j += 4) {
                        uint32_t*      h_k   = reinterpret_cast<uint32_t*>(k_loc + i + j);
                        uint32_t const i8s_k = reinterpret_cast<uint32_t const&>(*(k_quant + i + j));
                        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[0]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_k[1]) : "r"(i8s_k), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[0]) : "r"(h_k[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_k[1]) : "r"(h_k[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < K_LOAD_STRIDE; j += 2) {
                        k_loc_reordered[(k_context_group + i + j) * 2]     = k_loc[i + j]     * k_scale_loc[(i + j) / QUANT_GROUP];
                        k_loc_reordered[(k_context_group + i + j) * 2 + 1] = k_loc[i + j + 1] * k_scale_loc[(i + j + 1) / QUANT_GROUP];
                    }
                }
            } else {
                uint32_t* h_k = reinterpret_cast<uint32_t*>(k_loc_reordered);
                #pragma unroll
                for (int64_t i = 0; i < K_PER_CONTEXT / 2; i++) {
                    h_k[k_context_group + i * 2] = 0;
                }
            }
        }

        #pragma unroll
        for (int64_t i = 0; i < HEAD_SIZE / MMA_K; i++) {
            uint32_t* A = reinterpret_cast<uint32_t*>(k_loc_reordered + i * MMA_KPT);
            uint32_t* B = reinterpret_cast<uint32_t*>(q_loc + i * MMA_QPT);
            uint32_t* D = reinterpret_cast<uint32_t*>(qk_dot);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
                    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
            );
        }

        #pragma unroll
        for (int64_t i = 0; i < VALID_LOGIT_SIZE; i++) {
            const int64_t query_group_idx = group_lane_id + i * MMA_TPG;
            tile_qk_max[i] = -FLT_MAX;
            #pragma unroll
            for (int64_t j = 0; j < MMA_GPT; j++) {
                const int64_t l_context_id = base_id + group_id + j * MMA_GPW;
                if (l_context_id < block_context_len) {
                    qk_dot[j * MMA_LPG + i] *= p.attn_scale;
                    if (ATTN_MASK) {
                        qk_dot[j * MMA_LPG + i] += __half2float(attn_mask[query_group_idx * p.mask_stride_h + l_context_id]);
                    }
                    tile_qk_max[i] = fmaxf(tile_qk_max[i], qk_dot[j * MMA_LPG + i]);
                }
            }
            #pragma unroll
            for (int32_t mask = MMA_TPG; mask < WARP_SIZE; mask <<= 1) {
                tile_qk_max[i] = fmaxf(tile_qk_max[i], __shfl_xor_sync(uint32_t(-1), tile_qk_max[i], mask));
            }
            if (tile_qk_max[i] > warp_qk_max[i]) {
                fp32_t logit_scale = exp(warp_qk_max[i] - tile_qk_max[i]);
                #pragma unroll
                for (int64_t j = 0; j < O_SIZE; j += MMA_OPT) {
                    #pragma unroll
                    for (int64_t k = 0; k < MMA_GPT; k++) {
                        warp_o[j + k * MMA_OPG + i] *= logit_scale;
                    }
                }
                warp_exp_sum[i] *= logit_scale;
                warp_qk_max[i] = tile_qk_max[i];
            }
            #pragma unroll
            for (int64_t j = 0; j < MMA_GPT; j++) {
                const int64_t l_context_id = base_id + group_id + j * MMA_GPW;
                if (l_context_id < block_context_len) {
                    qk_dot[j * MMA_LPG + i] = exp(qk_dot[j * MMA_LPG + i] - warp_qk_max[i]);
                    warp_exp_sum[i] += qk_dot[j * MMA_LPG + i];
                }
            }
        }

        fp16_t logit[L_SIZE];
        attn_logits_reorder<L_SIZE>(logit, qk_dot, logit_reorder_id1, logit_reorder_id2);

        #pragma unroll
        for (int64_t v_context_group = 0; v_context_group < MMA_VPG; v_context_group++) {
            int64_t v_context_id = base_id + v_context_group * MMA_TPG + group_lane_id;
            if (v_context_id < block_context_len) {
                int8_t v_quant[V_PER_CONTEXT];
                fp16_t v_scale_loc[VS_PER_CONTEXT], v_scale_loc2[VS_PER_CONTEXT * 2], v_loc[V_PER_CONTEXT];
                const int64_t cache_token_idx = (PAGE_SIZE <= 0)
                                ? v_context_id
                                : (p.cachestarts[batch_idx * p.cachestarts_stride_b + (block_context_beg + v_context_id) / PAGE_SIZE]
                                    + ((block_context_beg + v_context_id) % PAGE_SIZE));
                const int64_t value_offset = cache_token_idx * p.cache_stride_s;

                #pragma unroll
                for (int64_t i = 0; i < VS_PER_CONTEXT; i += VS_LOAD_STRIDE) {
                    copy<sizeof(fp16_t) * VS_LOAD_STRIDE>(&v_scale_glb[value_offset / QUANT_GROUP + i],  &v_scale_loc[i]);
                }

                if (V_PER_CONTEXT % QUANT_GROUP) {
                    if (group_id % 2) {
                        #pragma unroll
                        for (int64_t i = 1; i < VS_PER_CONTEXT; i++) {
                            v_scale_loc2[2 * i] = v_scale_loc[i];
                            v_scale_loc2[2 * i - 1] = v_scale_loc[i];
                        }
                        v_scale_loc2[0] = v_scale_loc[0];
                    } else {
                        #pragma unroll
                        for (int64_t i = 0; i < VS_PER_CONTEXT - 1; i++) {
                            v_scale_loc2[2 * i] = v_scale_loc[i];
                            v_scale_loc2[2 * i + 1] = v_scale_loc[i];
                        }
                        v_scale_loc2[2 * VS_PER_CONTEXT - 2] = v_scale_loc[VS_PER_CONTEXT - 1];
                    }
                }

                #pragma unroll
                for (int64_t i = 0; i < V_PER_CONTEXT; i += V_LOAD_STRIDE) {
                    const int64_t value_idx = value_offset + i;
                    copy<sizeof(int8_t) * V_LOAD_STRIDE>(&v_glb[value_idx],  &v_quant[i]);

                    #pragma unroll
                    for (int64_t j = 0; j < V_LOAD_STRIDE; j++) {
                        v_quant[i + j] += 128;
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < V_LOAD_STRIDE; j += 4) {
                        uint32_t*      h_v   = reinterpret_cast<uint32_t*>(v_loc + i + j);
                        uint32_t const i8s_v = reinterpret_cast<uint32_t const&>(*(v_quant + i + j));
                        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[0]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
                        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h_v[1]) : "r"(i8s_v), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
                        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[0]) : "r"(h_v[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
                        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h_v[1]) : "r"(h_v[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < V_LOAD_STRIDE; j += 2) {
                        fp16_t v_scale = (V_PER_CONTEXT % QUANT_GROUP)
                                    ? v_scale_loc2[(i + j) / (QUANT_GROUP / 2)]
                                    : v_scale_loc[(i + j) / QUANT_GROUP];
                        v_loc_reordered[2 * v_context_group - v_context_group % 2 + (i + j) * MMA_VPG]     = v_loc[i + j]     * v_scale;
                        v_loc_reordered[2 * v_context_group - v_context_group % 2 + (i + j) * MMA_VPG + 2] = v_loc[i + j + 1] * v_scale;
                    }
                }
            } else {
                uint16_t* h_v = reinterpret_cast<uint16_t*>(v_loc_reordered);
                #pragma unroll
                for (int64_t i = 0; i < V_PER_CONTEXT; i += MMA_GPT) {
                    h_v[2 * v_context_group - v_context_group % 2 + i * MMA_VPG]     = 0;
                    h_v[2 * v_context_group - v_context_group % 2 + i * MMA_VPG + 2] = 0;
                }
            }
        }

        #pragma unroll
        for (int64_t i = 0; i < HEAD_SIZE / MMA_M; i++) {
            uint32_t* A = reinterpret_cast<uint32_t*>(v_loc_reordered + i * MMA_VPT);
            uint32_t* B = reinterpret_cast<uint32_t*>(logit);
            uint32_t* D = reinterpret_cast<uint32_t*>(warp_o + i * MMA_OPT);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
                    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1])
            );
        }
    }

    fp32_t *reduce_buffer = reinterpret_cast<fp32_t*>(q_shm + (HEAD_SIZE + F16PV) * FULL_GROUP_SIZE);
    fp32_t block_qk_max[MMA_LPG], warp_o_reordered[O_SIZE];
    attn_block_reduce_group_max<WPT, MMA_TPG, VALID_LOGIT_SIZE>(block_qk_max, warp_qk_max, reduce_buffer);

    #pragma unroll
    for (int64_t i = 0; i < VALID_LOGIT_SIZE; i++) {
        if (block_qk_max[i] > warp_qk_max[i]) {
            fp32_t logit_scale = exp(warp_qk_max[i] - block_qk_max[i]);
            warp_exp_sum[i] *= logit_scale;
            #pragma unroll
            for (int64_t j = 0; j < O_SIZE; j += MMA_LPG) {
                warp_o[i + j] *= logit_scale;
            }
        }
    }

    attn_block_reduce_group_sum<WPT, MMA_TPG, VALID_LOGIT_SIZE>(warp_exp_sum, &reduce_buffer[WPT * FULL_GROUP_SIZE]);

    if (MULTI_BLOCK > 1 && tid < MMA_TPG) {
        if (is_tail_group) {
            #pragma unroll
            for (int64_t i = 0; i < MMA_LPG; i++) {
                int64_t head_offset_loc = group_lane_id + i * MMA_TPG;
                if (head_offset_loc < TAIL_GROUP_SIZE) {
                    partial_log_sum[head_offset_loc * num_batchs * MULTI_BLOCK + block_idx]
                        = block_qk_max[i] + log(warp_exp_sum[i]);
                }
            }
        } else {
            #pragma unroll
            for (int64_t i = 0; i < MMA_LPG; i++) {
                partial_log_sum[(group_lane_id * MMA_LPG + i) * num_batchs * MULTI_BLOCK + block_idx]
                    = block_qk_max[i] + log(warp_exp_sum[i]);
            }
        }
    }

    #pragma unroll
    for (int64_t i = 0; i < VALID_LOGIT_SIZE; i++) {
        const fp32_t inv_sum = __fdividef(1.f, warp_exp_sum[i] + 1e-6f);
        #pragma unroll
        for (int64_t j = 0; j < O_SIZE / MMA_LPG; j++) {
            warp_o_reordered[i * O_SIZE / MMA_LPG + j] = warp_o[i + j * MMA_LPG] * inv_sum;
        }
    }

    const int64_t reduce_warp_id  = tid % WPT;

    #pragma unroll
    for (int64_t head_reduce = 0; head_reduce < VALID_REDUCE_HEAD; head_reduce += HEAD_PER_REDUCE) {
        // wait for logits to be reused
        __syncthreads();

        if (is_tail_group) {
            #pragma unroll
            for (int64_t i = 0; i < VALID_LOGIT_SIZE; i++) {
                int64_t head_offset_glb = group_lane_id + i * MMA_TPG;
                if (head_offset_glb >= head_reduce && head_offset_glb <= head_reduce + HEAD_PER_REDUCE && head_offset_glb < TAIL_GROUP_SIZE) {
                    int64_t head_offset_shm = head_offset_glb - head_reduce;

                    #pragma unroll
                    for (int64_t j = 0; j < O_PER_QUERY; j += F32PV) {
                        copy<sizeof(fp32_t) * F32PV>(
                            &warp_o_reordered[i * O_PER_QUERY + j],
                            &tmp_buffer[
                                head_offset_shm * (WPT * HEAD_SIZE + F32PV)
                                + group_id * O_PER_QUERY * WPT
                                + j * WPT
                                + warp_id * F32PV]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int64_t i = 0; i < HEAD_PER_REDUCE / FULL_GROUP_SIZE + 1; i++) {
                if (
                    (HEAD_PER_REDUCE == 8)
                    || (HEAD_PER_REDUCE == 4)
                    || (HEAD_PER_REDUCE == 2 && (group_lane_id == head_reduce % 4 || group_lane_id == head_reduce % 4 + 1))
                    || (HEAD_PER_REDUCE == 1 && group_lane_id == head_reduce % 4)
                ) {
                    int64_t head_offset_shm
                                = (HEAD_PER_REDUCE == 8) ? (group_lane_id + i * MMA_TPG)
                                : (HEAD_PER_REDUCE == 4) ? group_lane_id
                                : (HEAD_PER_REDUCE == 2) ? (group_lane_id % HEAD_PER_REDUCE)
                                : 0;
                    int64_t head_offset_loc = (HEAD_PER_REDUCE == 8) ? i : (head_reduce / 4);

                    #pragma unroll
                    for (int64_t j = 0; j < O_PER_QUERY; j += F32PV) {
                        copy<sizeof(fp32_t) * F32PV>(
                            &warp_o_reordered[head_offset_loc * O_PER_QUERY + j],
                            &tmp_buffer[
                                head_offset_shm * (WPT * HEAD_SIZE + F32PV)
                                + group_id * O_PER_QUERY * WPT
                                + j * WPT
                                + warp_id * F32PV]);
                    }
                }
            }
        }

        __syncthreads();
        #pragma unroll
        for (int64_t i = 0; i < THREAD_PER_REDUCE; i += TPB) {
            fp16_t block_o_loc[F16PV];

            const int64_t head_dim_idx      = (i + tid) / WPT % THREAD_PER_HEAD * F16PV;
            const int64_t head_offset_shm   = (i + tid) / WPT / THREAD_PER_HEAD;
            const int64_t head_offset_glb   = head_offset_shm + head_reduce;

            bool is_reduce_thread = is_tail_group
                            ? (head_offset_glb < TAIL_GROUP_SIZE)
                            : (THREAD_PER_REDUCE % TPB == 0 || head_offset_shm < HEAD_PER_REDUCE);

            if (is_reduce_thread) {
                #pragma unroll
                for (int64_t j = 0; j < F16PV; j += F32PV) {
                    copy<sizeof(fp32_t) * F32PV>(
                        &tmp_buffer[
                            head_offset_shm * (WPT * HEAD_SIZE + F32PV)
                            + head_dim_idx * WPT
                            + j * WPT
                            + reduce_warp_id * F32PV],
                        &warp_o[j]);
                }
            }

            #pragma unroll
            for (int64_t j = 0; j < F16PV; j++) {
                #pragma unroll
                for (int32_t mask = WPT / 2; mask > 0; mask >>= 1) {
                    warp_o[j] += __shfl_xor_sync(uint32_t(-1), warp_o[j], mask);
                }
                block_o_loc[j] = __float2half(warp_o[j]);
            }

            if (reduce_warp_id == 0 && is_reduce_thread) {
                fp16_t *block_o_glb = (MULTI_BLOCK == 1)
                            ? &o_glb[head_offset_glb * HEAD_SIZE + head_dim_idx]
                            : &partial_o[
                                head_offset_glb * num_batchs * HEAD_SIZE * MULTI_BLOCK
                                + head_dim_idx * MULTI_BLOCK
                                + block_idx * F16PV];
                copy<sizeof(fp16_t) * F16PV>(block_o_loc, block_o_glb);
            }
        }
    }

    // Flash decoding
    if (MULTI_BLOCK > 1) {
        __syncthreads();

        bool last_block = false;
        // Make sure every block finishs the partial computation.
        if (tid == 0) {
            if (atomicAdd(block_counter, 1) == MULTI_BLOCK - 1) {
                last_block = true;
            }
        }

        // The last block do the final computation.
        if (__syncthreads_or(last_block)) {
            const int64_t multi_block_idx   = tid % MULTI_BLOCK;
            const int64_t head_dim_base     = tid / MULTI_BLOCK * F16PV;
            const int64_t head_dim_stride   = TPB / MULTI_BLOCK * F16PV;

            #pragma unroll
            for (int64_t head_offset = 0; head_offset < VALID_REDUCE_HEAD; head_offset++) {
                if ((QUERY_GROUP > FULL_GROUP_SIZE)
                        && (TAIL_GROUP_SIZE > 0)
                        && (blockIdx.x % GROUP_BLOCK_SIZE == GROUP_BLOCK_SIZE - 1)
                        && (head_offset >= TAIL_GROUP_SIZE)){
                    break;
                }

                // get max block log sum exp
                fp32_t local_log_sum_exp = (warp_lane_id < MULTI_BLOCK)
                            ? partial_log_sum[head_offset * num_batchs * MULTI_BLOCK + multi_block_idx]
                            : -FLT_MAX;
                fp32_t max_log_sum_exp = local_log_sum_exp;
                # pragma unroll
                for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                    max_log_sum_exp = fmaxf(max_log_sum_exp, __shfl_xor_sync(uint32_t(-1), max_log_sum_exp, mask));
                }
                max_log_sum_exp = __shfl_sync(uint32_t(-1), max_log_sum_exp, 0);

                // update scale
                fp32_t local_scale = (warp_lane_id < MULTI_BLOCK)
                            ? exp(local_log_sum_exp - max_log_sum_exp)
                            : 0.f;
                fp32_t scale_sum = local_scale;
                # pragma unroll
                for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                    scale_sum += __shfl_xor_sync(uint32_t(-1), scale_sum, mask);
                }
                scale_sum = __shfl_sync(uint32_t(-1), scale_sum, 0);

                int scale_id = warp_id * MULTI_BLOCK + warp_lane_id;
                if (warp_lane_id < MULTI_BLOCK && scale_id < WARP_SIZE) {
                    tmp_buffer[scale_id] = local_scale / scale_sum;
                }
                __syncthreads();

                // block ouput reduce
                for (int64_t head_dim_idx = head_dim_base; head_dim_idx < HEAD_SIZE; head_dim_idx += head_dim_stride) {
                    fp16_t final_out[F16PV];
                    local_scale = tmp_buffer[warp_lane_id];
                    copy<sizeof(fp16_t) * F16PV>(
                        &partial_o[
                            head_offset * num_batchs * HEAD_SIZE * MULTI_BLOCK +
                            head_dim_idx * MULTI_BLOCK +
                            multi_block_idx * F16PV],
                        final_out);

                    #pragma unroll
                    for (int64_t i = 0; i < F16PV; i++) {
                        fp32_t float_out = __half2float(final_out[i]) * local_scale;
                        # pragma unroll
                        for (int32_t mask = MULTI_BLOCK / 2; mask >= 1; mask /= 2) {
                            float_out += __shfl_xor_sync(uint32_t(-1), float_out, mask);
                        }
                        final_out[i] = __float2half(float_out);
                    }

                    if (multi_block_idx == 0) {
                        copy<sizeof(fp16_t) * F16PV>(
                            final_out,
                            &o_glb[head_offset * HEAD_SIZE + head_dim_idx]);
                    }
                }
            }
        }
    }
}


#include "cuda_runtime_api.h"
#include "cuda_runtime.h"

#include <iostream>
#include <cmath>
#include "cuptr.hpp"

#include <fstream>
#include "nlohmann/json.hpp"
using json = nlohmann::json;


struct MHCAConfig
{
    dynamic_batching_decoding_cache_attention_kernel_param mhca_p;

    int32_t perf;
    int32_t kernel;
    int32_t tpb;
    int32_t multi_block;
    int32_t attn_mask;
    int32_t head_size;
    int32_t query_group;
    int32_t num_batchs;
    int32_t num_heads;
    int32_t seq_len;
    int32_t shm_size;

    MHCAConfig(const char* jsonfile=nullptr) {
        if(jsonfile != nullptr)
            loadFromJson(jsonfile);

    }

    void loadFromJson(const char* jsonfile)
    {
        std::ifstream f(jsonfile);
        json j = json::parse(f);

        mhca_p.attn_scale = j["attn_scale"].get<float>();
        mhca_p.layer_idx = j["layer_idx"].get<int64_t>();
        mhca_p.num_kv_repeats = j["query_group"].get<int64_t>();
        mhca_p.query_stride_s = j["query_stride_s"].get<int64_t>();
        mhca_p.output_stride_s = j["output_stride_s"].get<int64_t>();
        mhca_p.mask_stride_s = j["mask_stride_s"].get<int64_t>();
        mhca_p.mask_stride_h = j["mask_stride_h"].get<int64_t>();
        mhca_p.cache_stride_s = j["cache_stride_s"].get<int64_t>();
        mhca_p.cache_stride_l = j["cache_stride_l"].get<int64_t>();
        mhca_p.cache_stride_h = j["cache_stride_h"].get<int64_t>();
        mhca_p.cache_stride_kv = j["cache_stride_kv"].get<int64_t>();

        perf = j["perf"].get<int32_t>();
        kernel = j["kernel"].get<int32_t>();
        tpb = j["tpb"].get<int32_t>();
        multi_block = j["multi_block"].get<int32_t>();
        attn_mask = j["attn_mask"].get<int32_t>();
        head_size = j["head_size"].get<int32_t>();
        num_batchs = j["num_batchs"].get<int32_t>();
        num_heads = j["num_heads"].get<int32_t>();
        seq_len = j["seq_len"].get<int32_t>();
        query_group = j["query_group"].get<int32_t>();

        if (multi_block == 1) {
            multi_block = 32 / (head_size / 64);
        } else {
            multi_block = 1;
        }

        shm_size = 0;
        if (kernel == 0) {
            int32_t wpt = tpb / 32;
            int32_t reduce_size = wpt * sizeof(float);
            int32_t kvlen = (seq_len * sizeof(float) + multi_block - 1) / multi_block;
            int32_t logits_size = max(kvlen, wpt * head_size * (int)(sizeof(float)));
            shm_size = reduce_size + logits_size;
        }
    }
};

template<int32_t HEAD_SIZE, int32_t TPB, bool ATTN_MASK, bool DO_MULTI_BLOCK>
int32_t test_mhca_mask_head_size(
    const cudaStream_t stream,
    MHCAConfig &cfg
) {
    const int32_t QUANT_GROUP = 8;
    const int32_t MULTI_BLOCK = DO_MULTI_BLOCK ? 32 / (HEAD_SIZE / 64) : 1;
    const int32_t THREAD_GROUP_SIZE = HEAD_SIZE / 64 * 4;

    auto kernel_fn = dynamic_batching_decoding_cache_sharemem_attention_fp16_kernel<HEAD_SIZE, THREAD_GROUP_SIZE, TPB, QUANT_GROUP, MULTI_BLOCK, ATTN_MASK, 0>;
    if (cfg.kernel == 1) {
        kernel_fn = dynamic_batching_decoding_cache_infinity_attention_fp16_kernel<HEAD_SIZE, THREAD_GROUP_SIZE, TPB, QUANT_GROUP, MULTI_BLOCK, ATTN_MASK, 0>;
    } else if (cfg.kernel == 2) {
        switch (cfg.query_group) {
            case 2:
                kernel_fn = dynamic_batching_decoding_group_query_cache_attention_fp16_kernel<HEAD_SIZE, TPB, QUANT_GROUP, 2, MULTI_BLOCK, ATTN_MASK, 0>;
                break;
            case 4:
                kernel_fn = dynamic_batching_decoding_group_query_cache_attention_fp16_kernel<HEAD_SIZE, TPB, QUANT_GROUP, 4, MULTI_BLOCK, ATTN_MASK, 0>;
                break;
            case 6:
                kernel_fn = dynamic_batching_decoding_group_query_cache_attention_fp16_kernel<HEAD_SIZE, TPB, QUANT_GROUP, 6, MULTI_BLOCK, ATTN_MASK, 0>;
                break;
            case 8:
                kernel_fn = dynamic_batching_decoding_group_query_cache_attention_fp16_kernel<HEAD_SIZE, TPB, QUANT_GROUP, 8, MULTI_BLOCK, ATTN_MASK, 0>;
                break;
            case 16:
                kernel_fn = dynamic_batching_decoding_group_query_cache_attention_fp16_kernel<HEAD_SIZE, TPB, QUANT_GROUP, 16, MULTI_BLOCK, ATTN_MASK, 0>;
                break;
            default:
                return 1;
        }
    } else if (cfg.kernel != 0) {
        return 1;
    }

    int64_t query_group = cfg.kernel == 2
                ? (cfg.query_group + 7) / 8 * (cfg.num_heads / cfg.query_group)
                : cfg.num_heads;
    const dim3 grid_size = {
        (unsigned int)query_group,
        (unsigned int)cfg.num_batchs,
        (unsigned int)cfg.multi_block};
    kernel_fn<<<grid_size, TPB, cfg.shm_size, stream>>>(cfg.mhca_p);

    return 0;
}

template<int32_t TPB, bool DO_MULTI_BLOCK>
int32_t test_mhca_tpb_multi_block(
    const cudaStream_t stream,
    MHCAConfig &cfg
) {
    int32_t res = 0;
    if (cfg.attn_mask == 1) {
        switch (cfg.head_size) {
            case 64:
                res = test_mhca_mask_head_size<64, TPB, true, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 96:
                res = test_mhca_mask_head_size<96, TPB, true, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 128:
                res = test_mhca_mask_head_size<128, TPB, true, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 256:
                res = test_mhca_mask_head_size<256, TPB, true, DO_MULTI_BLOCK>(stream, cfg);
                break;
            default:
                res = 1;
        }
    } else {
        switch (cfg.head_size) {
            case 64:
                res = test_mhca_mask_head_size<64, TPB, false, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 96:
                res = test_mhca_mask_head_size<96, TPB, false, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 128:
                res = test_mhca_mask_head_size<128, TPB, false, DO_MULTI_BLOCK>(stream, cfg);
                break;
            case 256:
                res = test_mhca_mask_head_size<256, TPB, false, DO_MULTI_BLOCK>(stream, cfg);
                break;
            default:
                res = 1;
        }
    }
    return res;
}

int32_t test_mhca(
    const cudaStream_t stream,
    MHCAConfig &cfg
) {
    int32_t res = 0;
    if (cfg.multi_block > 1) {
        switch (cfg.tpb) {
            case 256:
                res = test_mhca_tpb_multi_block<256, true>(stream, cfg);
                break;
            case 512:
                res = test_mhca_tpb_multi_block<512, true>(stream, cfg);
                break;
            case 1024:
                res = test_mhca_tpb_multi_block<1024, true>(stream, cfg);
                break;
            default:
                res = 1;
        }
    } else {
        switch (cfg.tpb) {
            case 256:
                res = test_mhca_tpb_multi_block<256, false>(stream, cfg);
                break;
            case 512:
                res = test_mhca_tpb_multi_block<512, false>(stream, cfg);
                break;
            case 1024:
                res = test_mhca_tpb_multi_block<1024, false>(stream, cfg);
                break;
            default:
                res = 1;
        }
    }
    return res;
}

void dotest_decoding_mhca() {
    MHCAConfig cfg("./data/ut_cfg_mhca.json");

    int64_t q_size = cfg.num_batchs * cfg.num_heads * cfg.head_size;
    int64_t mask_size = cfg.attn_mask == 0 ? 0 : cfg.num_batchs * cfg.num_heads * cfg.seq_len;
    int64_t o_size = cfg.num_batchs * cfg.num_heads * cfg.head_size;
    int64_t cache_size = cfg.num_batchs * cfg.seq_len * cfg.num_heads / cfg.query_group * cfg.head_size * 2;
    int64_t scale_size = cache_size / 8;
    int64_t cachestart_size = cfg.num_batchs;
    int64_t kvstart_size = cfg.num_batchs + 1;

    int64_t counter_size = cfg.multi_block > 0 ? cfg.num_batchs * cfg.num_heads : 0;
    int64_t log_sum_size = cfg.multi_block * cfg.num_batchs * cfg.num_heads;
    int64_t partial_out_size = cfg.multi_block * cfg.num_batchs * cfg.num_heads * cfg.head_size;

    HostPtr<fp16_t> h_q(q_size);
    HostPtr<int8_t> h_cache(cache_size);
    HostPtr<fp16_t> h_scale(scale_size);
    HostPtr<int64_t> h_cachestart(cachestart_size);
    HostPtr<int64_t> h_kvstart(kvstart_size);

    h_q.LoadFromFile("./data/ut_q.dat");
    h_cache.LoadFromFile("./data/ut_cache.dat");
    h_scale.LoadFromFile("./data/ut_scale.dat");
    h_cachestart.LoadFromFile("./data/ut_cache_starts.dat");
    h_kvstart.LoadFromFile("./data/ut_kv_starts.dat");

    CuPtr<fp16_t> d_q(h_q);
    CuPtr<fp16_t> d_o(o_size);
    CuPtr<int8_t> d_cache(h_cache);
    CuPtr<fp16_t> d_scale(h_scale);
    CuPtr<int64_t> d_cachestart(h_cachestart);
    CuPtr<int64_t> d_kvstart(h_kvstart);

    void* p_q = d_q.GetPtr();
    void* p_o = d_o.GetPtr();
    void* p_cache = d_cache.GetPtr();
    void* p_scale = d_scale.GetPtr();
    void* p_cachestart = d_cachestart.GetPtr();
    void* p_kvstart = d_kvstart.GetPtr();

    CuPtr<fp16_t> d_mask;
    void* p_mask = nullptr;
    if (cfg.attn_mask > 0) {
        HostPtr<fp16_t> h_mask(mask_size);
        h_mask.LoadFromFile("./data/ut_mask.dat");

        d_mask.Reset(h_mask);
        p_mask = d_mask.GetPtr();
    }

    cfg.mhca_p.query = (half*)p_q;
    cfg.mhca_p.attn_mask = (half*)p_mask;
    cfg.mhca_p.output = (half*)p_o;
    cfg.mhca_p.cache = (int8_t*)p_cache;
    cfg.mhca_p.scale = (half*)p_scale;
    cfg.mhca_p.cachestarts = (int64_t*)p_cachestart;
    cfg.mhca_p.kvstarts = (int64_t*)p_kvstart;

    CuPtr<int32_t> d_block_counter(counter_size);
    CuPtr<fp32_t> d_log_sum(log_sum_size);
    CuPtr<fp16_t> d_partial_out(partial_out_size);
    if (cfg.multi_block > 1) {
        void* p_counter = d_block_counter.GetPtr();
        void* p_log_sum = d_log_sum.GetPtr();
        void* p_partial_out = d_partial_out.GetPtr();

        cfg.mhca_p.multi_block.block_counter = (int32_t*)p_counter;
        cfg.mhca_p.multi_block.log_sum_exp = (fp32_t*)p_log_sum;
        cfg.mhca_p.multi_block.partial_out = (fp16_t*)p_partial_out;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int32_t res = test_mhca(stream, cfg);

    if (res == 1) {
        printf("Unsupport parameter.");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s", cudaGetErrorString(err));
        res = 1;
    }

    if (res == 0) {
        HostPtr<fp16_t> h_o;
        d_o.ToHostPtr(h_o);
        h_o.SaveToFile("./data/ut_ocmp.dat");
    }

    if (cfg.perf == 1 && res == 0) {
        constexpr int NWarmUp = 10;
        constexpr int NTest = 100;

        cudaEvent_t event_start, event_stop;
        checkCudaErrors(cudaEventCreate(&event_start));
        checkCudaErrors(cudaEventCreate(&event_stop));

        float elapsedTime;
        float totaltime = 0;
        checkCudaErrors(cudaEventRecord(event_start, stream));
        for(int itest=0; itest<NWarmUp; itest++)
        {
            test_mhca(stream, cfg);
        }
        checkCudaErrors(cudaEventRecord(event_stop, stream));
        checkCudaErrors(cudaEventSynchronize(event_stop));
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));
        
        elapsedTime = 0.0f;

        checkCudaErrors(cudaEventRecord(event_start, stream));
        for(int itest=0; itest<NTest; itest++)
        {
            test_mhca(stream, cfg);
        }
        checkCudaErrors(cudaEventRecord(event_stop, stream));
        checkCudaErrors(cudaEventSynchronize(event_stop));
        checkCudaErrors(cudaEventElapsedTime(&elapsedTime, event_start, event_stop));
        totaltime += elapsedTime;
        totaltime /= NTest;
        printf("Time: %10.4f ms.\n", totaltime);
        checkCudaErrors(cudaEventDestroy(event_start));
        checkCudaErrors(cudaEventDestroy(event_stop));
    }
}


int main() {
    dotest_decoding_mhca();
    return 0;
}
