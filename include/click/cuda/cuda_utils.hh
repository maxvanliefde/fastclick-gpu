#ifndef CUDA_UTILS_HH
#define CUDA_UTILS_HH

#include <cuda.h>
#include <cuda_runtime.h>

// inspired from nv-l2fwd
#define CUDA_CHECK(stmt, errh)                                              \
    do {                                                                    \
        cudaError_t result = (stmt);                                        \
        if(cudaSuccess != result) {                                         \
            errh->error("[%s:%d] cuda failed with %s \n",            \
                   __FILE__, __LINE__, cudaGetErrorString(result));         \
        }                                                                   \
    } while(0)

#endif