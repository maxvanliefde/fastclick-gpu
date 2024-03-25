#ifndef CUDA_UTILS_HH
#define CUDA_UTILS_HH

#include <cuda.h>
#include <cuda_runtime.h>

/* Max number of items in the GPU communication list */
#define MAX_BURSTS_X_QUEUE 4096

// inspired from nv-l2fwd
#define CUDA_CHECK(stmt, errh)                                              \
    do {                                                                    \
        cudaError_t result = (stmt);                                        \
        if(cudaSuccess != result) {                                         \
            return errh->error("[%s:%d] cuda failed with %s \n",            \
                   __FILE__, __LINE__, cudaGetErrorString(result));         \
        }                                                                   \
    } while(0)

#endif