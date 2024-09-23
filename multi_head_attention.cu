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

template <typename scalar_t>
__global__ void attention_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ out,
    scalar_t* __restrict__ grad_q,
    scalar_t* __restrict__ grad_k,
    scalar_t* __restrict__ grad_v,
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

    scalar_t q_val = q[qkv_offset + tid];
    scalar_t k_val = k[kv_seq_offset + seq_id * head_dim + tid];
    scalar_t v_val = v[kv_seq_offset + seq_id * head_dim + tid];
    scalar_t out_val = out[qkv_offset + tid];
    scalar_t grad_out_val = grad_out[qkv_offset + tid];

    // Placeholder for actual gradient computations dL/dS dL/dQ, dL/dK, dL/dV

    scalar_t grad_s = grad_out_val * v_val; // Placeholder

    scalar_t grad_q_val = grad_s * k_val / sqrtf((float)head_dim);
    scalar_t grad_k_val = grad_s * q_val / sqrtf((float)head_dim);
    scalar_t grad_v_val = grad_out_val * 1.0f; // Placeholder

    atomicAdd(&grad_q[qkv_offset + tid], grad_q_val);
    atomicAdd(&grad_k[kv_seq_offset + seq_id * head_dim + tid], grad_k_val);
    atomicAdd(&grad_v[kv_seq_offset + seq_id * head_dim + tid], grad_v_val);
}

torch::Tensor multi_head_attention_forward(
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

std::vector<torch::Tensor> multi_head_attention_backward(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out
) {
    const auto batch_size = q.size(0);
    const auto num_heads = q.size(1);
    const auto seq_len = q.size(2);
    const auto head_dim = q.size(3);

    auto grad_q = torch::zeros_like(q);
    auto grad_k = torch::zeros_like(k);
    auto grad_v = torch::zeros_like(v);

    const dim3 threads(head_dim);
    const dim3 blocks(seq_len, num_heads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "attention_backward_kernel", ([&] {
        attention_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            grad_q.data_ptr<scalar_t>(),
            grad_k.data_ptr<scalar_t>(),
            grad_v.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            num_heads,
            head_dim
        );
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "attention_backward_kernel launch failed");
    return {grad_q, grad_k, grad_v};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_attention_forward", &multi_head_attention_forward, "Multi-head attention forward");
    m.def("multi_head_attention_backward", &multi_head_attention_backward, "Multi-head attention backward");
}