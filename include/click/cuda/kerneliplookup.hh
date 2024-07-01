#ifndef CLICK_CUDA_IP_HH
#define CLICK_CUDA_IP_HH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rte_gpudev.h>
#include <click/cuda/routegpu.hh>

#ifdef HAVE_CUDA

void wrapper_ip_lookup(struct rte_gpu_comm_list *comm_list_item, int cuda_threads, RouteGPU *ip_list, uint32_t len);
void wrapper_ip_lookup_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len);


#endif /* HAVE_CUDA */

#endif /* CLICK_CUDA_IP_HH */