#include <cuda_runtime.h>

#include <cstdio>

namespace dsa_paged_ns {
#include "../../../../kernels/operators/dsa_paged/dsa_paged_final.cu"
}  // namespace dsa_paged_ns

namespace gdn_ns {
#include "../../../../kernels/operators/gdn/gdn_final.cu"
}  // namespace gdn_ns

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(err__));             \
            return 1;                                                      \
        }                                                                  \
    } while (0)

int main() {
    dsa_paged_ns::launch_dsa_paged_optimized();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gdn_ns::launch_gdn_optimized();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("{\"status\":\"ok\",\"kernels\":[\"dsa_paged\",\"gdn\"]}\n");
    return 0;
}
