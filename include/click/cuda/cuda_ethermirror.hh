#ifndef CLICK_CUDA_ETHERMIRROR_H
#define CLICK_CUDA_ETHERMIRROR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rte_gpudev.h>

#ifdef HAVE_CUDA

void wrapper_ether_mirror(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream);

#endif /* HAVE_CUDA */

#endif /* CLICK_CUDA_ETHERMIRROR_H */