#include <stdio.h>

#include <click/cuda/kernelethermirror.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__global__ void kernel_ether_mirror(struct rte_gpu_comm_list *comm_list_item) {
    int pkt_id = threadIdx.x;

    struct rte_ether_hdr *eth;
    uint8_t *src, *dst, tmp[6];
    size_t size;

    /* Workload */
    if (pkt_id < comm_list_item[0].num_pkts && comm_list_item[0].pkt_list[pkt_id].addr != NULL) {
        eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list_item[0].pkt_list[pkt_id].addr)));
        size = comm_list_item[0].pkt_list[pkt_id].size;
        src = (uint8_t *) (&eth->src_addr);
        dst = (uint8_t *) (&eth->dst_addr);


        /* Verify source and dest of ethernet addresses */
        // printf("Before Swap, Size: %lu, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     size,
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);

        /* Swap addresses */
        uint8_t j;
        for (j = 0; j < 6; j++) tmp[j] = src[j];
        for (j = 0; j < 6; j++) src[j] = dst[j];
        for (j = 0; j < 6; j++) dst[j] = tmp[j];

        /* Verify source and dest of ethernet addresses */
        // printf("After Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
    }

    /* Finish batch */
    __threadfence();
    __syncthreads();

    if (pkt_id == 0) {
        RTE_GPU_VOLATILE(*comm_list_item[0].status_d) = RTE_GPU_COMM_LIST_DONE;
        __threadfence_system(); // ensures writes are seen by the CPU
    }

    __syncthreads();
}

void wrapper_ether_mirror(struct rte_gpu_comm_list *comm_list_item, int cuda_threads) {
    kernel_ether_mirror <<< 1, cuda_threads >>> (comm_list_item);
}


__global__ void kernel_ether_mirror_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size) {
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
            eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list[item_id].pkt_list[pkt_id].addr)));
            size = comm_list[item_id].pkt_list[pkt_id].size;
            src = (uint8_t *) (&eth->src_addr);
            dst = (uint8_t *) (&eth->dst_addr);


            /* Verify source and dest of ethernet addresses */
            // printf("Before Swap, Size: %lu, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
            //     size,
            //     src[0], src[1], src[2], src[3], src[4], src[5], 
            //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);

            /* Swap addresses */
            uint8_t j;
            for (j = 0; j < 6; j++) tmp[j] = src[j];
            for (j = 0; j < 6; j++) src[j] = dst[j];
            for (j = 0; j < 6; j++) dst[j] = tmp[j];

            /* Verify source and dest of ethernet addresses */
            // printf("After Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
            //     src[0], src[1], src[2], src[3], src[4], src[5], 
            //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
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

void wrapper_ether_mirror_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads, cudaStream_t stream) {
    kernel_ether_mirror_persistent <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, comm_list_size);
}

#endif /* HAVE_CUDA */