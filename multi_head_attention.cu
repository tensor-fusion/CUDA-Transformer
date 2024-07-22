#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;

    __shared__ scalar_t s_k[32][33];
    __shared__ scalar_t s_v[32][33];

    const scalar_t scale = rsqrtf(head_dim);

    for (int seq_id = 0; seq_id < seq_len; seq_id++) {
        scalar_t q_val = 0;
        if (tid < head_dim) {
            q_val = q[((batch_id * num_heads + head_id) * seq_len + seq_id) * head_dim + tid] * scale;
        }

        scalar_t max_score = -INFINITY;
        scalar_t sum_exp = 0;
        scalar_t out_val = 0;

        for (int i = 0; i < seq_len; i += 32) {
            if (tid < head_dim && (i + tid) < seq_len) {
                s_k[tid][threadIdx.y] = k[((batch_id * num_heads + head_id) * seq_len + i + tid) * head_dim + threadIdx.y];
                s_v[tid][threadIdx.y] = v[((batch_id * num_heads + head_id) * seq_len + i + tid) * head_dim + threadIdx.y];
            }
            __syncthreads();

            for (int j = 0; j < 32 && (i + j) < seq_len; ++j) {
                scalar_t score = q_val * s_k[j][tid];
                max_score = max(max_score, score);
            }
            __syncthreads();
        }

        // softmax
        for (int i = 0; i < seq_len; i += 32) {
            if (tid < head_dim && (i + tid) < seq_len) {
                s_k[tid][threadIdx.y] = k[((batch_id * num_heads + head_id) * seq_len + i + tid) * head_dim + threadIdx.y];
                s_v[tid][threadIdx.y] = v[((batch_id * num_heads + head_id) * seq_len + i + tid) * head_dim + threadIdx.y];
            }
            __syncthreads();

            for (int j = 0; j < 32 && (i + j) < seq_len; ++j) {
                scalar_t score = q_val * s_k[j][tid];
                scalar_t exp_score = __expf(score - max_score);
                sum_exp += exp_score;
                out_val += exp_score * s_v[j][tid];
            }
            __syncthreads();
        }

        // norm
        if (tid < head_dim) {
            out_val /= sum_exp;
            out[((batch_id * num_heads + head_id) * seq_len + seq_id) * head_dim + tid] = out_val;
        }
    }
}

torch::Tensor multi_head_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    const auto batch_size = q.size(0);
    const auto seq_len = q.size(1);
    const auto num_heads = q.size(2);
    const auto head_dim = q.size(3);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    auto out = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);

    const dim3 threads(32, 32);
    const dim3 blocks(1, num_heads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.type(), "attention_kernel", ([&] {
        attention_kernel<scalar_t><<<blocks, threads>>>(
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

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_attention", &multi_head_attention_cuda, "Multi-head attention (CUDA)");
}
