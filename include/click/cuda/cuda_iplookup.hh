#ifndef CLICK_CUDA_IPLOOKUP_H
#define CLICK_CUDA_IPLOOKUP_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rte_gpudev.h>
#include <click/cuda/routegpu.hh>

#ifdef HAVE_CUDA

void wrapper_iplookup(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len);

#endif /* HAVE_CUDA */

#endif /* CLICK_CUDA_IPLOOKUP_H */