// ===========================================================================
// RMSNorm (hidden_size = 4096, bfloat16) —— 官方 baseline 语义的 CUDA 直译起点
//
// 说明：
//   1. 本文件用于新的“官方 baseline 派生”闭环起点。
//   2. 语义直接对齐数据集 solutions/baseline/.../flashinfer_wrapper_2e27cd.json
//   3. 后续所有新轮次必须以本目录中的官方 baseline 源码为派生依据，不再从旧的自研 round0 起步。
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
constexpr float kEpsilon = 1.0e-6f;

__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ y,
                                    int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* x_row = x + static_cast<size_t>(row) * hidden;
    __nv_bfloat16* y_row = y + static_cast<size_t>(row) * hidden;

    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);
    const int hidden2 = hidden >> 1;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden2; i += kThreadsPerBlock) {
        __nv_bfloat162 v = x_row2[i];
        float a = __bfloat162float(v.x);
        float b = __bfloat162float(v.y);
        local_sum += a * a + b * b;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }

    __shared__ float warp_sum[kThreadsPerBlock / 32];
    if ((tid & 31) == 0) {
        warp_sum[tid >> 5] = local_sum;
    }
    __syncthreads();

    float row_sum = 0.0f;
    if (tid < 32) {
        row_sum = (tid < (kThreadsPerBlock / 32)) ? warp_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_down_sync(0xFFFFFFFFu, row_sum, offset);
        }
        if (tid == 0) {
            warp_sum[0] = row_sum;
        }
    }
    __syncthreads();

    const float inv_rms =
        rsqrtf(warp_sum[0] / static_cast<float>(hidden) + kEpsilon);

    const __nv_bfloat162* w_row2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* y_row2 = reinterpret_cast<__nv_bfloat162*>(y_row);
    for (int i = tid; i < hidden2; i += kThreadsPerBlock) {
        __nv_bfloat162 xv = x_row2[i];
        __nv_bfloat162 wv = w_row2[i];

        float x0 = __bfloat162float(xv.x);
        float x1 = __bfloat162float(xv.y);
        float w0 = __bfloat162float(wv.x);
        float w1 = __bfloat162float(wv.y);

        __nv_bfloat162 out;
        out.x = __float2bfloat16_rn(x0 * inv_rms * w0);
        out.y = __float2bfloat16_rn(x1 * inv_rms * w1);
        y_row2[i] = out;
    }
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
    TVM_FFI_ICHECK_EQ(hidden % 2, 0) << "hidden 必须为偶数（向量化要求）";

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

    dim3 grid(static_cast<unsigned int>(batch));
    dim3 block(kThreadsPerBlock);
    rmsnorm_bf16_kernel<<<grid, block, 0, stream>>>(
        x_ptr, w_ptr, y_ptr, static_cast<int>(hidden)
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_h4096, rmsnorm_h4096);
