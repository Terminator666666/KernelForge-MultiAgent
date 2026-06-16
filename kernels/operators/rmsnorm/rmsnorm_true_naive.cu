#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int RMSNORM_NAIVE_DEFAULT_HIDDEN_SIZE = 4096;
constexpr float RMSNORM_NAIVE_DEFAULT_EPS = 1.0e-6f;

__global__ void rmsnorm_true_naive_kernel(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    float rms_inv = rsqrtf(sum_sq / hidden_size + eps);

    for (int i = 0; i < hidden_size; i++) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        row_output[i] = __float2half(val * rms_inv * w);
    }
}

}  // namespace

void launch_rmsnorm_true_naive(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    rmsnorm_true_naive_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, batch_size, RMSNORM_NAIVE_DEFAULT_HIDDEN_SIZE, RMSNORM_NAIVE_DEFAULT_EPS
    );
}

void launch_rmsnorm_true_naive_ex(
    const half* input,
    half* output,
    const half* weight,
    int batch_size,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    rmsnorm_true_naive_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, batch_size, hidden_size, eps
    );
}
