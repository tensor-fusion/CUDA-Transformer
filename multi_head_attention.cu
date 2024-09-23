#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

template <typename scalar_t>
__global__ void attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int batch_id = blockIdx.z;
    const int head_id = blockIdx.y;
    const int seq_id = blockIdx.x;

    const int tid = threadIdx.x;

    if (seq_id >= seq_len || tid >= head_dim) return;

    const int qkv_offset = ((batch_id * num_heads + head_id) * seq_len + seq_id) * head_dim;
    const int kv_seq_offset = (batch_id * num_heads + head_id) * seq_len * head_dim;

    scalar_t q_value = q[qkv_offset + tid];

    extern __shared__ float shared_mem[];
    float* s_scores = shared_mem;
    float* s_max_scores = s_scores + seq_len;
    float* s_sum_exp = s_max_scores + 1;

    if (tid == 0) {
        s_max_scores[0] = -INFINITY;
        s_sum_exp[0] = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            s_scores[i] = 0.0f;
        }
    }
    __syncthreads();

    // Attention scores
    for (int i = 0; i < seq_len; ++i) {
        scalar_t k_value = k[kv_seq_offset + i * head_dim + tid];
        float score = static_cast<float>(q_value) * static_cast<float>(k_value);

        atomicAdd(&s_scores[i], score);
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < seq_len; ++i) {
            s_scores[i] /= sqrtf((float)head_dim);
            s_max_scores[0] = fmaxf(s_max_scores[0], s_scores[i]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < seq_len; ++i) {
            s_scores[i] = __expf(s_scores[i] - s_max_scores[0]);
            s_sum_exp[0] += s_scores[i];
        }
    }
    __syncthreads();

    float out_value = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        scalar_t v_value = v[kv_seq_offset + i * head_dim + tid];
        float weight = s_scores[i] / s_sum_exp[0];
        out_value += weight * static_cast<float>(v_value);
    }

    out[qkv_offset + tid] = static_cast<scalar_t>(out_value);
}

torch::Tensor multi_head_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    auto out = torch::empty({batch_size, num_heads, seq_len, head_dim}, options);

    const dim3 threads(head_dim);
    const dim3 blocks(seq_len, num_heads, batch_size);

    const size_t shared_mem_size = (seq_len + 2) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_kernel", ([&] {
        attention_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            num_heads,
            head_dim
        );
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "attention_kernel launch failed");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_attention", &multi_head_attention_cuda, "Multi-head attention");
}