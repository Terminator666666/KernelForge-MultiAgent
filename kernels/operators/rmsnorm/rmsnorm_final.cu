#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kRmsNormBlockSize = 256;
constexpr int kRmsNormWarpSize = 32;
constexpr int kRmsNormWarpCount = kRmsNormBlockSize / kRmsNormWarpSize;
constexpr int kRmsNormDefaultHiddenSize = 4096;
constexpr float kRmsNormDefaultEps = 1.0e-6f;

struct WorkspaceCache {
    int device = -1;
    float* partial_sums = nullptr;
    float* inv_rms = nullptr;
    size_t partial_capacity = 0;
    size_t inv_capacity = 0;

    ~WorkspaceCache() {
        if (partial_sums != nullptr) {
            cudaFree(partial_sums);
        }
        if (inv_rms != nullptr) {
            cudaFree(inv_rms);
        }
    }
};

thread_local WorkspaceCache g_workspace_cache;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kRmsNormWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
    const int lane = threadIdx.x & (kRmsNormWarpSize - 1);
    const int warp_id = threadIdx.x / kRmsNormWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < kRmsNormWarpCount) ? shared[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared[0] = block_sum;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float half2_sum_sq(half2 value) {
    const float2 as_float = __half22float2(value);
    return as_float.x * as_float.x + as_float.y * as_float.y;
}

__global__ void rmsnorm_single_pass_half2_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    int batch_size,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    const int hidden_size_vec = hidden_size >> 1;
    const half2* row_input = reinterpret_cast<const half2*>(input + static_cast<size_t>(row) * hidden_size);
    const half2* row_weight = reinterpret_cast<const half2*>(weight);
    half2* row_output = reinterpret_cast<half2*>(output + static_cast<size_t>(row) * hidden_size);

    __shared__ float shared[kRmsNormWarpCount];
    float partial_sum = 0.0f;

    for (int i = threadIdx.x; i < hidden_size_vec; i += blockDim.x) {
        partial_sum += half2_sum_sq(row_input[i]);
    }

    const float total_sum = block_reduce_sum(partial_sum, shared);
    const float inv_rms = rsqrtf(total_sum / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size_vec; i += blockDim.x) {
        const float2 input_value = __half22float2(row_input[i]);
        const float2 weight_value = __half22float2(row_weight[i]);
        row_output[i] = __floats2half2_rn(
            input_value.x * inv_rms * weight_value.x,
            input_value.y * inv_rms * weight_value.y
        );
    }
}

__global__ void rmsnorm_single_pass_scalar_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    int batch_size,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    const half* row_input = input + static_cast<size_t>(row) * hidden_size;
    half* row_output = output + static_cast<size_t>(row) * hidden_size;

    __shared__ float shared[kRmsNormWarpCount];
    float partial_sum = 0.0f;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float value = __half2float(row_input[i]);
        partial_sum += value * value;
    }

    const float total_sum = block_reduce_sum(partial_sum, shared);
    const float inv_rms = rsqrtf(total_sum / static_cast<float>(hidden_size) + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        const float value = __half2float(row_input[i]);
        const float scale = __half2float(weight[i]);
        row_output[i] = __float2half_rn(value * inv_rms * scale);
    }
}

__global__ void rmsnorm_partial_sum_half2_kernel(
    const half* __restrict__ input,
    float* __restrict__ partial_sums,
    int batch_size,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch_size) {
        return;
    }

    const int hidden_size_vec = hidden_size >> 1;
    const int split_stride = blockDim.x * gridDim.y;
    const half2* row_input = reinterpret_cast<const half2*>(input + static_cast<size_t>(row) * hidden_size);

    __shared__ float shared[kRmsNormWarpCount];
    float partial_sum = 0.0f;

    for (int i = split * blockDim.x + threadIdx.x; i < hidden_size_vec; i += split_stride) {
        partial_sum += half2_sum_sq(row_input[i]);
    }

    if ((hidden_size & 1) != 0 && split == 0 && threadIdx.x == 0) {
        const float tail = __half2float(input[static_cast<size_t>(row) * hidden_size + hidden_size - 1]);
        partial_sum += tail * tail;
    }

    const float reduced = block_reduce_sum(partial_sum, shared);
    if (threadIdx.x == 0) {
        partial_sums[static_cast<size_t>(row) * gridDim.y + split] = reduced;
    }
}

__global__ void rmsnorm_partial_sum_scalar_kernel(
    const half* __restrict__ input,
    float* __restrict__ partial_sums,
    int batch_size,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch_size) {
        return;
    }

    const int split_stride = blockDim.x * gridDim.y;
    const half* row_input = input + static_cast<size_t>(row) * hidden_size;

    __shared__ float shared[kRmsNormWarpCount];
    float partial_sum = 0.0f;

    for (int i = split * blockDim.x + threadIdx.x; i < hidden_size; i += split_stride) {
        const float value = __half2float(row_input[i]);
        partial_sum += value * value;
    }

    const float reduced = block_reduce_sum(partial_sum, shared);
    if (threadIdx.x == 0) {
        partial_sums[static_cast<size_t>(row) * gridDim.y + split] = reduced;
    }
}

__global__ void rmsnorm_finalize_inv_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ inv_rms,
    int batch_size,
    int hidden_size,
    int split_count,
    float eps
) {
    const int row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }

    __shared__ float shared[kRmsNormWarpCount];
    float partial_sum = 0.0f;

    for (int i = threadIdx.x; i < split_count; i += blockDim.x) {
        partial_sum += partial_sums[static_cast<size_t>(row) * split_count + i];
    }

    const float total_sum = block_reduce_sum(partial_sum, shared);
    if (threadIdx.x == 0) {
        inv_rms[row] = rsqrtf(total_sum / static_cast<float>(hidden_size) + eps);
    }
}

__global__ void rmsnorm_apply_half2_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int batch_size,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch_size) {
        return;
    }

    const int hidden_size_vec = hidden_size >> 1;
    const int split_stride = blockDim.x * gridDim.y;
    const float row_inv_rms = inv_rms[row];

    const half2* row_input = reinterpret_cast<const half2*>(input + static_cast<size_t>(row) * hidden_size);
    const half2* row_weight = reinterpret_cast<const half2*>(weight);
    half2* row_output = reinterpret_cast<half2*>(output + static_cast<size_t>(row) * hidden_size);

    for (int i = split * blockDim.x + threadIdx.x; i < hidden_size_vec; i += split_stride) {
        const float2 input_value = __half22float2(row_input[i]);
        const float2 weight_value = __half22float2(row_weight[i]);
        row_output[i] = __floats2half2_rn(
            input_value.x * row_inv_rms * weight_value.x,
            input_value.y * row_inv_rms * weight_value.y
        );
    }

    if ((hidden_size & 1) != 0 && split == 0 && threadIdx.x == 0) {
        const int tail = hidden_size - 1;
        const float input_value = __half2float(input[static_cast<size_t>(row) * hidden_size + tail]);
        const float weight_value = __half2float(weight[tail]);
        output[static_cast<size_t>(row) * hidden_size + tail] =
            __float2half_rn(input_value * row_inv_rms * weight_value);
    }
}

__global__ void rmsnorm_apply_scalar_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const float* __restrict__ inv_rms,
    int batch_size,
    int hidden_size
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch_size) {
        return;
    }

    const int split_stride = blockDim.x * gridDim.y;
    const float row_inv_rms = inv_rms[row];
    const half* row_input = input + static_cast<size_t>(row) * hidden_size;
    half* row_output = output + static_cast<size_t>(row) * hidden_size;

    for (int i = split * blockDim.x + threadIdx.x; i < hidden_size; i += split_stride) {
        const float input_value = __half2float(row_input[i]);
        const float weight_value = __half2float(weight[i]);
        row_output[i] = __float2half_rn(input_value * row_inv_rms * weight_value);
    }
}

int choose_split_count(int batch_size, int hidden_size) {
    if (hidden_size < 1024) {
        return 1;
    }
    if (batch_size <= 8) {
        return 8;
    }
    if (batch_size <= 32) {
        return 4;
    }
    if (batch_size <= 128) {
        return 2;
    }
    return 1;
}

bool ensure_workspace(size_t partial_count, size_t inv_count) {
    int current_device = 0;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        return false;
    }

    WorkspaceCache& cache = g_workspace_cache;
    if (cache.device != current_device) {
        if (cache.partial_sums != nullptr) {
            cudaFree(cache.partial_sums);
        }
        if (cache.inv_rms != nullptr) {
            cudaFree(cache.inv_rms);
        }
        cache = WorkspaceCache{};
        cache.device = current_device;
    }

    if (cache.partial_capacity < partial_count) {
        if (cache.partial_sums != nullptr) {
            cudaFree(cache.partial_sums);
            cache.partial_sums = nullptr;
        }
        if (cudaMalloc(&cache.partial_sums, partial_count * sizeof(float)) != cudaSuccess) {
            cache.partial_capacity = 0;
            return false;
        }
        cache.partial_capacity = partial_count;
    }

    if (cache.inv_capacity < inv_count) {
        if (cache.inv_rms != nullptr) {
            cudaFree(cache.inv_rms);
            cache.inv_rms = nullptr;
        }
        if (cudaMalloc(&cache.inv_rms, inv_count * sizeof(float)) != cudaSuccess) {
            cache.inv_capacity = 0;
            return false;
        }
        cache.inv_capacity = inv_count;
    }

    return true;
}

void launch_single_pass(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const bool use_half2 = (hidden_size % 2) == 0;
    if (use_half2) {
        rmsnorm_single_pass_half2_kernel<<<batch_size, kRmsNormBlockSize, 0, stream>>>(
            input, output, weight, batch_size, hidden_size, eps
        );
    } else {
        rmsnorm_single_pass_scalar_kernel<<<batch_size, kRmsNormBlockSize, 0, stream>>>(
            input, output, weight, batch_size, hidden_size, eps
        );
    }
}

void launch_split_path(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    int split_count,
    float eps,
    cudaStream_t stream
) {
    if (!ensure_workspace(static_cast<size_t>(batch_size) * split_count, batch_size)) {
        launch_single_pass(input, output, weight, batch_size, hidden_size, eps, stream);
        return;
    }

    WorkspaceCache& cache = g_workspace_cache;
    dim3 partial_grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(split_count));
    dim3 finalize_grid(static_cast<unsigned int>(batch_size));
    const bool use_half2 = (hidden_size % 2) == 0;

    if (use_half2) {
        rmsnorm_partial_sum_half2_kernel<<<partial_grid, kRmsNormBlockSize, 0, stream>>>(
            input, cache.partial_sums, batch_size, hidden_size
        );
    } else {
        rmsnorm_partial_sum_scalar_kernel<<<partial_grid, kRmsNormBlockSize, 0, stream>>>(
            input, cache.partial_sums, batch_size, hidden_size
        );
    }

    rmsnorm_finalize_inv_kernel<<<finalize_grid, kRmsNormBlockSize, 0, stream>>>(
        cache.partial_sums, cache.inv_rms, batch_size, hidden_size, split_count, eps
    );

    if (use_half2) {
        rmsnorm_apply_half2_kernel<<<partial_grid, kRmsNormBlockSize, 0, stream>>>(
            input, output, weight, cache.inv_rms, batch_size, hidden_size
        );
    } else {
        rmsnorm_apply_scalar_kernel<<<partial_grid, kRmsNormBlockSize, 0, stream>>>(
            input, output, weight, cache.inv_rms, batch_size, hidden_size
        );
    }
}

}  // namespace

void launch_rmsnorm_optimized_ex(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
);

void launch_rmsnorm_optimized(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
);

void launch_rmsnorm_optimized(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    cudaStream_t stream
) {
    launch_rmsnorm_optimized(
        input,
        output,
        weight,
        batch_size,
        kRmsNormDefaultHiddenSize,
        stream
    );
}

void launch_rmsnorm_optimized(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
) {
    launch_rmsnorm_optimized_ex(
        input,
        output,
        weight,
        batch_size,
        hidden_size,
        kRmsNormDefaultEps,
        stream
    );
}

void launch_rmsnorm_optimized_ex(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    if (input == nullptr || output == nullptr || weight == nullptr) {
        return;
    }
    if (batch_size <= 0 || hidden_size <= 0) {
        return;
    }

    const int split_count = choose_split_count(batch_size, hidden_size);
    if (split_count <= 1) {
        launch_single_pass(input, output, weight, batch_size, hidden_size, eps, stream);
        return;
    }

    launch_split_path(input, output, weight, batch_size, hidden_size, split_count, eps, stream);
}
