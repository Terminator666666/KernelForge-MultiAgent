// Convenience translation unit for building the current operator set.
// For production builds, compile only the individual operator files you need.

#include "../operators/softmax/softmax_final.cu"
#include "../operators/softmax/softmax_true_naive.cu"
#include "../operators/matmul/matmul_final.cu"
#include "../operators/matmul/matmul_true_naive.cu"
#include "../operators/layernorm/layernorm_final.cu"
#include "../operators/layernorm/layernorm_true_naive.cu"
#include "../operators/rmsnorm/rmsnorm_final.cu"
#include "../operators/rmsnorm/rmsnorm_true_naive.cu"
