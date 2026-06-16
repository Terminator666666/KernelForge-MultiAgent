#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Shared Memory per Block (static): %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Dynamic Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlockOptin);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    
    return 0;
}
