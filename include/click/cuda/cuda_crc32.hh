#ifndef CLICK_CUDA_CRC32_HH
#define CLICK_CUDA_CRC32_HH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <rte_gpudev.h>


#ifdef HAVE_CUDA
void wrapper_setcrc32_commlist(struct rte_gpu_comm_list *comm_list_item, uint32_t *crc_table, int cuda_blocks, int cuda_threads, cudaStream_t stream);
void wrapper_setcrc32_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, uint32_t *crc_table, int cuda_blocks, int cuda_threads, cudaStream_t stream);
void wrapper_crc32_coalescent(char *batch_memory, uint32_t n_pkts, uint32_t pkt_size, uint32_t *crc_table, int blocks, int threads, cudaStream_t stream);

#endif /* HAVE_CUDA */

#endif /* CLICK_CUDA_CRC32_HH */