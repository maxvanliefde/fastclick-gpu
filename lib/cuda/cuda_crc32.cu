#include <stdio.h>

#include <click/cuda/cuda_crc32.hh>
#include <rte_ether.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cub/block/block_scan.cuh>

using namespace cooperative_groups;

#ifdef HAVE_CUDA

/* COMMUNICATION LIST */

__global__ void kernel_setcrc32_commlist(struct rte_gpu_comm_list *comm_list_item, uint32_t *crc_table) {
    int pkt_id = threadIdx.x;

    struct rte_ether_hdr *eth;
    uint8_t *src, *dst, tmp[6];
    size_t size;


    /* Workload */
    if (pkt_id < comm_list_item->num_pkts && comm_list_item->pkt_list[pkt_id].addr != NULL) {
        size = comm_list_item->pkt_list[pkt_id].size - RTE_ETHER_CRC_LEN;

        int i, j;
        uint32_t crc_accum = 0xffffffff;
        char *data = (char *) comm_list_item->pkt_list[pkt_id].addr;
        for (j = 0;  j < size;  j++ ) {
            i = ( (uint32_t) ( crc_accum >> 24) ^ *data++ ) & 0xff;
            crc_accum = ( crc_accum << 8 ) ^ crc_table[i];
        }

        memcpy((void *) (comm_list_item->pkt_list[pkt_id].addr + size), &crc_accum, RTE_ETHER_CRC_LEN);

        /* EtherMirror */
        eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list_item[0].pkt_list[pkt_id].addr)));
        src = (uint8_t *) (&eth->src_addr);
        dst = (uint8_t *) (&eth->dst_addr);

        uint8_t k;
        for (k = 0; k < 6; k++) tmp[k] = src[k];
        for (k = 0; k < 6; k++) src[k] = dst[k];
        for (k = 0; k < 6; k++) dst[k] = tmp[k];
    }

    /* Finish batch */
    __threadfence();
    __syncthreads();

    if (pkt_id == 0) {
        RTE_GPU_VOLATILE(*comm_list_item->status_d) = RTE_GPU_COMM_LIST_DONE;
        __threadfence_system(); // ensures writes are seen by the CPU
    }
}

void wrapper_setcrc32_commlist(struct rte_gpu_comm_list *comm_list_item, uint32_t *crc_table, int cuda_blocks, int cuda_threads, cudaStream_t stream) {
    kernel_setcrc32_commlist <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list_item, crc_table);
} 


__global__ void kernel_setcrc32_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, uint32_t *crc_table) {
    int pkt_id = threadIdx.x;
    uint32_t item_id = blockIdx.x;
    
    struct rte_ether_hdr *eth;
    uint8_t *src, *dst, tmp[6];
    size_t size;

    /* Used for synchronization */
    enum rte_gpu_comm_list_status wait_status;
    __shared__ enum rte_gpu_comm_list_status wait_status_shared;
    __syncthreads();

    for(;;) {
        /* check status of current batch */
        if (pkt_id == 0) {
            while(1) {
                wait_status = RTE_GPU_VOLATILE(*comm_list[item_id].status_d);
                if (wait_status != RTE_GPU_COMM_LIST_FREE) {
                    wait_status_shared = wait_status;
                    __threadfence_block();
                    break;
                }
            }
        }

        __syncthreads();        // ensures all threads in the thread block have reached this point and all memory accesses made are visible to all threads in the block. 
        __threadfence_system(); // ensures seeing the writes made by the CPU before

        if (wait_status_shared != RTE_GPU_COMM_LIST_READY) break;

        /* Workload */
        if (pkt_id < comm_list[item_id].num_pkts && comm_list[item_id].pkt_list[pkt_id].addr != NULL) {
            size = comm_list[item_id].pkt_list[pkt_id].size - RTE_ETHER_CRC_LEN;

            int i, j;
            uint32_t crc_accum = 0xffffffff;
            char *data = (char *) comm_list[item_id].pkt_list[pkt_id].addr;
            for (j = 0;  j < size;  j++ ) {
                i = ( (uint32_t) ( crc_accum >> 24) ^ *data++ ) & 0xff;
                crc_accum = ( crc_accum << 8 ) ^ crc_table[i];
            }

            memcpy((void *) (comm_list[item_id].pkt_list[pkt_id].addr + size), &crc_accum, RTE_ETHER_CRC_LEN);

            /* EtherMirror */
            eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list[item_id].pkt_list[pkt_id].addr)));
            src = (uint8_t *) (&eth->src_addr);
            dst = (uint8_t *) (&eth->dst_addr);

            uint8_t k;
            for (k = 0; k < 6; k++) tmp[k] = src[k];
            for (k = 0; k < 6; k++) src[k] = dst[k];
            for (k = 0; k < 6; k++) dst[k] = tmp[k];
        }

        /* Finish batch */
        __threadfence();
        __syncthreads();

        if (pkt_id == 0) {
            RTE_GPU_VOLATILE(*comm_list[item_id].status_d) = RTE_GPU_COMM_LIST_DONE;
            __threadfence_system(); // ensures writes are seen by the CPU
        }

        item_id = (item_id + gridDim.x) % comm_list_size;
    }
}

void wrapper_setcrc32_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, uint32_t *crc_table, int cuda_blocks, int cuda_threads, cudaStream_t stream) {
    kernel_setcrc32_persistent <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, comm_list_size, crc_table);
} 


/* COALESCENT */

__global__ void kernel_crc32_coalescent(char *batch_memory, uint32_t n_pkts, uint32_t max_pkt_size, uint32_t *crc_table) {
    int pkt_id = blockIdx.x * blockDim.x + threadIdx.x;

    /* Workload */
    if (pkt_id < n_pkts) {
        uint32_t pkt_size = ((uint32_t *) batch_memory)[pkt_id];
        uint32_t size = pkt_size - RTE_ETHER_CRC_LEN;
        char *data = batch_memory + (n_pkts * sizeof(uint32_t)) + (max_pkt_size * pkt_id);

        int i, j;
        uint32_t crc_accum = 0xffffffff;
        for (j = 0;  j < size;  j++ ) {
            i = ( (uint32_t) ( crc_accum >> 24) ^ *data++ ) & 0xff;
            crc_accum = ( crc_accum << 8 ) ^ crc_table[i];
        }

        char *crc_pkt = batch_memory + (n_pkts * (max_pkt_size + sizeof(uint32_t))) + (RTE_ETHER_CRC_LEN * pkt_id);

        memcpy((void *) (crc_pkt), &crc_accum, RTE_ETHER_CRC_LEN);
    }

    __syncthreads();
}

void wrapper_crc32_coalescent(char *batch_memory, uint32_t n_pkts, uint32_t max_pkt_size, uint32_t *crc_table, int blocks, int threads, cudaStream_t stream) {
    kernel_crc32_coalescent <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts, max_pkt_size, crc_table);
}


#endif /* HAVE_CUDA */