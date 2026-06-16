// ===========================================================================
// RMSNorm (hidden_size = 4096, bfloat16) —— FlashInfer-Bench TVM-FFI 适配实现
//
// 对应 definition: rmsnorm_h4096
//   inputs : hidden_states [batch_size, 4096] bfloat16
//            weight        [4096]             bfloat16
//   outputs: output        [batch_size, 4096] bfloat16
//   语义   : output = (x / sqrt(mean(x^2) + eps)) * weight，eps 固定 1e-5
//            归约与缩放在 float32 中完成，最后写回 bfloat16（与官方 PyTorch 参考一致）
//
// 绑定方式: tvm-ffi（框架无关，DLPack 互操作）
// 入口签名: rmsnorm_h4096(hidden_states, weight, output)  —— DPS（输出在末位）
//
// 设计要点（Round 0：正确性优先，同时做基础向量化）:
//   - 每个 block 处理一行（一个 batch 元素），256 线程
//   - 用 __nv_bfloat162 做 2 路向量化访存，提高带宽利用率
//   - 两级归约：warp 内 __shfl_down_sync + 共享内存跨 warp 归约
// ===========================================================================

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// 每个 block 的线程数（= 8 个 warp）
constexpr int kThreadsPerBlock = 256;
// RMSNorm 的数值稳定项，官方 definition 固定为 1e-5
constexpr float kEpsilon = 1.0e-5f;

// ---------------------------------------------------------------------------
// 核函数：每个 block 归一化一行
// hidden 必须为偶数（4096 满足），以便按 bfloat162 做 2 路向量化
// ---------------------------------------------------------------------------
__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ y,
                                     int hidden) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 当前行的起始指针
    const __nv_bfloat16* x_row = x + static_cast<size_t>(row) * hidden;
    __nv_bfloat16* y_row = y + static_cast<size_t>(row) * hidden;

    // 以 bfloat162 视角访问（每次处理 2 个元素）
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);
    const int hidden2 = hidden >> 1;  // 向量化后的元素个数

    // -----------------------------------------------------------------------
    // 1) 每线程局部平方和（float32 累加，保证精度）
    // -----------------------------------------------------------------------
    float local_sum = 0.0f;
    for (int i = tid; i < hidden2; i += kThreadsPerBlock) {
        __nv_bfloat162 v = x_row2[i];
        float a = __bfloat162float(v.x);
        float b = __bfloat162float(v.y);
        local_sum += a * a + b * b;
    }

    // -----------------------------------------------------------------------
    // 2) 块内归约：先 warp 内归约，再跨 warp 归约
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, offset);
    }

    // 每个 warp 的部分和写入共享内存（共 8 个 warp）
    __shared__ float warp_sum[kThreadsPerBlock / 32];
    if ((tid & 31) == 0) {
        warp_sum[tid >> 5] = local_sum;
    }
    __syncthreads();

    // 由第 0 个 warp 完成最终归约
    float row_sum = 0.0f;
    if (tid < 32) {
        row_sum = (tid < (kThreadsPerBlock / 32)) ? warp_sum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_down_sync(0xFFFFFFFFu, row_sum, offset);
        }
        if (tid == 0) {
            warp_sum[0] = row_sum;  // 最终平方和存回 warp_sum[0]
        }
    }
    __syncthreads();

    // inv_rms = 1 / sqrt(mean(x^2) + eps)
    const float inv_rms =
        rsqrtf(warp_sum[0] / static_cast<float>(hidden) + kEpsilon);

    // -----------------------------------------------------------------------
    // 3) 归一化并乘以权重，写回 bfloat16
    // -----------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// TVM-FFI 入口函数
// DPS 约定：输出张量 output 作为最后一个参数传入，原地写入
// ---------------------------------------------------------------------------
void rmsnorm_h4096(tvm::ffi::TensorView hidden_states,
                   tvm::ffi::TensorView weight,
                   tvm::ffi::TensorView output) {
    // ---- 形状校验 ----
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states 必须是 2D [batch, hidden]";
    TVM_FFI_ICHECK_EQ(weight.ndim(), 1) << "weight 必须是 1D [hidden]";
    TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output 必须是 2D [batch, hidden]";

    const int64_t batch = hidden_states.size(0);
    const int64_t hidden = hidden_states.size(1);

    TVM_FFI_ICHECK_EQ(weight.size(0), hidden) << "weight 长度必须等于 hidden";
    TVM_FFI_ICHECK_EQ(output.size(0), batch) << "output.batch 必须与输入一致";
    TVM_FFI_ICHECK_EQ(output.size(1), hidden) << "output.hidden 必须与输入一致";
    TVM_FFI_ICHECK_EQ(hidden % 2, 0) << "hidden 必须为偶数（向量化要求）";

    // ---- dtype 校验：bfloat16 (DLPack code = kDLBfloat, bits = 16) ----
    DLDataType dt = hidden_states.dtype();
    TVM_FFI_ICHECK(dt.code == kDLBfloat && dt.bits == 16)
        << "hidden_states 必须是 bfloat16";
    TVM_FFI_ICHECK(weight.dtype().code == kDLBfloat && weight.dtype().bits == 16)
        << "weight 必须是 bfloat16";
    TVM_FFI_ICHECK(output.dtype().code == kDLBfloat && output.dtype().bits == 16)
        << "output 必须是 bfloat16";

    // ---- 连续性与设备校验 ----
    TVM_FFI_ICHECK(hidden_states.IsContiguous()) << "hidden_states 必须连续";
    TVM_FFI_ICHECK(weight.IsContiguous()) << "weight 必须连续";
    TVM_FFI_ICHECK(output.IsContiguous()) << "output 必须连续";

    DLDevice dev = hidden_states.device();
    TVM_FFI_ICHECK_EQ(dev.device_type, kDLCUDA) << "张量必须在 CUDA 设备上";

    if (batch == 0) return;  // 空输入直接返回

    // ---- 取数据指针 ----
    const __nv_bfloat16* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
    const __nv_bfloat16* w_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    __nv_bfloat16* y_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    // ---- 取当前 CUDA stream（由 FlashInfer-Bench 运行时提供）----
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    // ---- 启动核函数：每行一个 block ----
    dim3 grid(static_cast<unsigned int>(batch));
    dim3 block(kThreadsPerBlock);
    rmsnorm_bf16_kernel<<<grid, block, 0, stream>>>(
        x_ptr, w_ptr, y_ptr, static_cast<int>(hidden));
}

// 导出给 Python 调用
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_h4096, rmsnorm_h4096);
