// ===========================================================================
// RMSNorm (hidden_size = 4096, bfloat16) —— FlashInfer-Bench TVM-FFI 闭环优化实现
//
// 对应 definition: rmsnorm_h4096
//   inputs : hidden_states [batch_size, 4096] bfloat16
//            weight        [4096]             bfloat16
//   outputs: output        [batch_size, 4096] bfloat16
//   语义   : output = (x / sqrt(mean(x^2) + eps)) * weight
//            其中 eps 固定为 1e-5，归约与缩放均在 float32 中完成
//
// Round 1 设计要点：
//   1. 保留 Round 0 的 bfloat162 向量化与 float32 累加
//   2. 增加“小 batch 分裂并行”路径，用 3 段式 kernel 提高 CTA 数量
//   3. 大 batch 继续走单阶段快路径，避免额外 kernel launch 开销
// ===========================================================================

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHiddenSize = 4096;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;
constexpr float kEpsilon = 1.0e-5f;

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
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* shared) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < kWarpCount) ? shared[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared[0] = block_sum;
        }
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float bf162_sum_sq(__nv_bfloat162 value) {
    const float x0 = __bfloat162float(value.x);
    const float x1 = __bfloat162float(value.y);
    return x0 * x0 + x1 * x1;
}

__global__ void rmsnorm_single_pass_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int hidden
) {
    const int row = blockIdx.x;
    const int hidden2 = hidden >> 1;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(
        x + static_cast<size_t>(row) * hidden
    );
    const __nv_bfloat162* w_row2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* y_row2 = reinterpret_cast<__nv_bfloat162*>(
        y + static_cast<size_t>(row) * hidden
    );

    __shared__ float shared[kWarpCount];
    float partial_sum = 0.0f;

    for (int i = threadIdx.x; i < hidden2; i += blockDim.x) {
        partial_sum += bf162_sum_sq(x_row2[i]);
    }

    const float total_sum = block_reduce_sum(partial_sum, shared);
    const float inv_rms = rsqrtf(total_sum / static_cast<float>(hidden) + kEpsilon);

    for (int i = threadIdx.x; i < hidden2; i += blockDim.x) {
        const __nv_bfloat162 xv = x_row2[i];
        const __nv_bfloat162 wv = w_row2[i];
        __nv_bfloat162 out;
        out.x = __float2bfloat16_rn(__bfloat162float(xv.x) * inv_rms * __bfloat162float(wv.x));
        out.y = __float2bfloat16_rn(__bfloat162float(xv.y) * inv_rms * __bfloat162float(wv.y));
        y_row2[i] = out;
    }
}

__global__ void rmsnorm_partial_sum_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ partial_sums,
    int batch,
    int hidden
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch) {
        return;
    }

    const int hidden2 = hidden >> 1;
    const int split_stride = blockDim.x * gridDim.y;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(
        x + static_cast<size_t>(row) * hidden
    );

    __shared__ float shared[kWarpCount];
    float partial_sum = 0.0f;

    for (int i = split * blockDim.x + threadIdx.x; i < hidden2; i += split_stride) {
        partial_sum += bf162_sum_sq(x_row2[i]);
    }

    const float reduced = block_reduce_sum(partial_sum, shared);
    if (threadIdx.x == 0) {
        partial_sums[static_cast<size_t>(row) * gridDim.y + split] = reduced;
    }
}

__global__ void rmsnorm_finalize_inv_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ inv_rms,
    int batch,
    int hidden,
    int split_count
) {
    const int row = blockIdx.x;
    if (row >= batch) {
        return;
    }

    __shared__ float shared[kWarpCount];
    float partial_sum = 0.0f;

    for (int i = threadIdx.x; i < split_count; i += blockDim.x) {
        partial_sum += partial_sums[static_cast<size_t>(row) * split_count + i];
    }

    const float total_sum = block_reduce_sum(partial_sum, shared);
    if (threadIdx.x == 0) {
        inv_rms[row] = rsqrtf(total_sum / static_cast<float>(hidden) + kEpsilon);
    }
}

__global__ void rmsnorm_apply_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    const float* __restrict__ inv_rms,
    int batch,
    int hidden
) {
    const int row = blockIdx.x;
    const int split = blockIdx.y;
    if (row >= batch) {
        return;
    }

    const int hidden2 = hidden >> 1;
    const int split_stride = blockDim.x * gridDim.y;
    const float row_inv_rms = inv_rms[row];

    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(
        x + static_cast<size_t>(row) * hidden
    );
    const __nv_bfloat162* w_row2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* y_row2 = reinterpret_cast<__nv_bfloat162*>(
        y + static_cast<size_t>(row) * hidden
    );

    for (int i = split * blockDim.x + threadIdx.x; i < hidden2; i += split_stride) {
        const __nv_bfloat162 xv = x_row2[i];
        const __nv_bfloat162 wv = w_row2[i];
        __nv_bfloat162 out;
        out.x = __float2bfloat16_rn(__bfloat162float(xv.x) * row_inv_rms * __bfloat162float(wv.x));
        out.y = __float2bfloat16_rn(__bfloat162float(xv.y) * row_inv_rms * __bfloat162float(wv.y));
        y_row2[i] = out;
    }
}

int choose_split_count(int batch) {
    if (batch <= 8) {
        return 8;
    }
    if (batch <= 32) {
        return 4;
    }
    if (batch <= 128) {
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
    const __nv_bfloat16* x_ptr,
    const __nv_bfloat16* w_ptr,
    __nv_bfloat16* y_ptr,
    int batch,
    int hidden,
    cudaStream_t stream
) {
    rmsnorm_single_pass_bf16_kernel<<<batch, kThreadsPerBlock, 0, stream>>>(
        x_ptr, w_ptr, y_ptr, hidden
    );
}

void launch_split_path(
    const __nv_bfloat16* x_ptr,
    const __nv_bfloat16* w_ptr,
    __nv_bfloat16* y_ptr,
    int batch,
    int hidden,
    int split_count,
    cudaStream_t stream
) {
    if (!ensure_workspace(static_cast<size_t>(batch) * split_count, batch)) {
        launch_single_pass(x_ptr, w_ptr, y_ptr, batch, hidden, stream);
        return;
    }

    WorkspaceCache& cache = g_workspace_cache;
    dim3 split_grid(static_cast<unsigned int>(batch), static_cast<unsigned int>(split_count));
    dim3 row_grid(static_cast<unsigned int>(batch));

    rmsnorm_partial_sum_bf16_kernel<<<split_grid, kThreadsPerBlock, 0, stream>>>(
        x_ptr, cache.partial_sums, batch, hidden
    );
    rmsnorm_finalize_inv_kernel<<<row_grid, kThreadsPerBlock, 0, stream>>>(
        cache.partial_sums, cache.inv_rms, batch, hidden, split_count
    );
    rmsnorm_apply_bf16_kernel<<<split_grid, kThreadsPerBlock, 0, stream>>>(
        x_ptr, w_ptr, y_ptr, cache.inv_rms, batch, hidden
    );
}

}  // namespace

void rmsnorm_h4096(tvm::ffi::TensorView hidden_states,
                   tvm::ffi::TensorView weight,
                   tvm::ffi::TensorView output) {
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states 必须是 2D [batch, hidden]";
    TVM_FFI_ICHECK_EQ(weight.ndim(), 1) << "weight 必须是 1D [hidden]";
    TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output 必须是 2D [batch, hidden]";

    const int64_t batch = hidden_states.size(0);
    const int64_t hidden = hidden_states.size(1);

    TVM_FFI_ICHECK_EQ(hidden, kHiddenSize) << "rmsnorm_h4096 仅支持 hidden=4096";
    TVM_FFI_ICHECK_EQ(weight.size(0), hidden) << "weight 长度必须等于 hidden";
    TVM_FFI_ICHECK_EQ(output.size(0), batch) << "output.batch 必须与输入一致";
    TVM_FFI_ICHECK_EQ(output.size(1), hidden) << "output.hidden 必须与输入一致";
    TVM_FFI_ICHECK_EQ(hidden % 2, 0) << "hidden 必须为偶数（bfloat162 向量化要求）";

    DLDataType dt = hidden_states.dtype();
    TVM_FFI_ICHECK(dt.code == kDLBfloat && dt.bits == 16)
        << "hidden_states 必须是 bfloat16";
    TVM_FFI_ICHECK(weight.dtype().code == kDLBfloat && weight.dtype().bits == 16)
        << "weight 必须是 bfloat16";
    TVM_FFI_ICHECK(output.dtype().code == kDLBfloat && output.dtype().bits == 16)
        << "output 必须是 bfloat16";

    TVM_FFI_ICHECK(hidden_states.IsContiguous()) << "hidden_states 必须连续";
    TVM_FFI_ICHECK(weight.IsContiguous()) << "weight 必须连续";
    TVM_FFI_ICHECK(output.IsContiguous()) << "output 必须连续";

    DLDevice dev = hidden_states.device();
    TVM_FFI_ICHECK_EQ(dev.device_type, kDLCUDA) << "张量必须在 CUDA 设备上";

    if (batch == 0) {
        return;
    }

    const __nv_bfloat16* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
    const __nv_bfloat16* w_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    __nv_bfloat16* y_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    const int split_count = choose_split_count(static_cast<int>(batch));
    if (split_count <= 1) {
        launch_single_pass(
            x_ptr,
            w_ptr,
            y_ptr,
            static_cast<int>(batch),
            static_cast<int>(hidden),
            stream
        );
        return;
    }

    launch_split_path(
        x_ptr,
        w_ptr,
        y_ptr,
        static_cast<int>(batch),
        static_cast<int>(hidden),
        split_count,
        stream
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_h4096, rmsnorm_h4096);
