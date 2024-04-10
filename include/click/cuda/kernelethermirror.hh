#ifndef CLICK_CUDA_HELLO_HH
#define CLICK_CUDA_HELLO_HH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rte_gpudev.h>

#ifdef HAVE_CUDA

void wrapper_ether_mirror(struct rte_gpu_comm_list *comm_list_item, int cuda_threads);
void wrapper_ether_mirror_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads, cudaStream_t stream);


#endif /* HAVE_CUDA */

#endif /* CLICK_CUDA_HELLO_HH */