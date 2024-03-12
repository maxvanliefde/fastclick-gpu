#include <stdio.h>

#include <click/cuda/kernelethermirror.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__global__ void kernel_ether_mirror(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = 0;

    struct rte_ether_hdr *eth;
    uint8_t *src, *dst, tmp[6];
    size_t size;

    /* Used for synchronization */
    enum rte_gpu_comm_list_status wait_status;
    __shared__ enum rte_gpu_comm_list_status wait_status_shared;
    __syncthreads();

    for(;;) {
        /* check status of current batch */
        if (idx == 0) {
            while(1) {
                wait_status = RTE_GPU_VOLATILE(*comm_list[i].status_d);
                if (wait_status != RTE_GPU_COMM_LIST_FREE) {
                    wait_status_shared = wait_status;
                    __threadfence_block();
                    break;
                }
            }
        }

        __syncthreads();

        if (wait_status_shared != RTE_GPU_COMM_LIST_READY) break;

        /* Workload */
        if (idx < comm_list[i].num_pkts && comm_list[i].pkt_list[idx].addr != NULL) {
            eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list[i].pkt_list[idx].addr)));
            size = comm_list[i].pkt_list[idx].size;
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

        if (idx == 0) {
            RTE_GPU_VOLATILE(*comm_list[i].status_d) = RTE_GPU_COMM_LIST_DONE;
            __threadfence_system();
        }

        i = (i+1) % comm_list_size;
    }
}

void wrapper_ether_mirror(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads) {
    kernel_ether_mirror <<< cuda_blocks, cuda_threads >>> (comm_list, comm_list_size);
}

#endif /* HAVE_CUDA */