//int B, int T, int C // 输入输出shape, 默认为 8, 1024, 768
//const float* inp // 输入x, shape为 [B, T, C]
//float* mean, float* rstd // 输入x的均值\miu, 标准差的倒数 1/\sigma
//const float* weight, const float* bias // 可学习的权重及偏置， 随机初始化后传入
//float* out // 输出, shape为 [B, T, C]

void layernorm_forward_cpu(float* out, float* mean, float* rstd, const float* inp, const float* weight, const float* bias, int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b, t, :]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i ++) {
                m += x[i];
            }
            m = m/C;
            //calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_forward1(float* out, float* mean, float* rstd, const float* inp, const float* weight, const float* bias, int B, int T, int C, const int block_size){
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size){
    extern __shard__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int  i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void normlization_kernel(float* out, const float* inp, float* mean, float* rstd, const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

__global__ void layernorm_forward3(float* __restrict__ out, float* __restrict__ rstd, const float* __restrict__ inp, const float* __restrict__ weight,
                                const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index  
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()){
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    //rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

//block 级别的 reduce
auto block = cg::this_thread_block(); // definitions

auto warp32 = cg::tiled_partition<32>(block); // 32 thread warps
auto warp16 = cg::tiled_partition<16>(block); // 16 thread tiles
auto warp8 = cg::tiled_partition<8>(block); // 8 thread tiles
auto tile8 = cg::tiled_partition<8>(warp32); // 8 thread sub-warps
auto tile4 = cg::tiled_partition<4>(tile8); // 4 thread sub-sub warps

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; //term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= B*T) return;
int b = idx / T;
int t = idx % T;

const float* dout_bt = dout + b * T * C + t * C;
const float* inp_bt = inp + b * T * C + t * C;
float* dinp_bt = dinp + b * T * C + t * C;
const float mean_bt = mean[b * T + t];
const float rstd_bt = rstd[b * T + t];

// first: two reduce operaions
float dnorm_mean = 0.0f;
float dnorm_norm_mean = 0.0f;
for (int i = 0; i < C; i++) {
    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
    float dnorm_i = weight[i] * dout_bt[i];
    dnorm_mean += dnorm_i;
    dnorm_norm_mean += dnorm_i * norm_bti;
}
dnorm_mean = dnorm_mean / C;
dnorm_norm_mean = dnorm_norm_mean / C;

// now iterate again and accumulate all the gradients
for (int i = 0; i < C; i++) {
    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
    float dnorm_i = weight[i] * dout_bt[i];
    // gradient contribution to bias
    atomicAdd(&dbias[i], dout_bt[i]);
    // gradient contribution to weight
    atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
    // gradient contribution to input
    float dval = 0.0f;
    dval += dnorm_i; // term 1
    dval -= dnorm_mean; // term 2
    dval -= norm_bti * dnorm_norm_mean; // term 3
    dval *= rstd_bt; // final scale
    dinp_bt[i] += dval;
}

extern __shared__ float shared[]; // size = 2 * C

namespace cg = cooperative_groups;
cg::thread_block block = cg::this_thread_block();
cg::thread_block_tile<32> warp = cg::this_thread_block();
int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
int N = B * T;
if(idx >= N) { return; } // thread guards

int b = idx / T;
int t = idx % T;

const float* dout_bt = dout + b * T * C + t * C;
const float* inp_bt = inp + b * T * C + t * C;
float* dinp_bt = dinp + b * T * C + t * C;
const float mean_bt = mean[b * T + t];
const float rstd_bt = rstd[b * T + t];

// the first half of shared memory is bias, second is weight
float* dbias_shared = shared;
float* dweight_shared = shared + C;

// init shared memory to zero
#pragma unroll
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
        for(int i = threadIdx.x; i < C; i += blockDim.x){
            atomicAdd(&dbias[i], dbias_shared[i]);
            atomicAdd(&dweight[i], dweight_shared[i]);
        }