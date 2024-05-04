#include <stdio.h>

#include <click/cuda/kerneliplookup.hh>
#include <rte_ether.h>
#include <rte_ip.h>

#ifdef HAVE_CUDA

__device__ uint8_t longest_match(uint32_t addr1, uint32_t addr2) {
    uint8_t len_match = 0;
    for (uint8_t j = 0; j < 4; j++){
        uint16_t mask = 7;
        for (int16_t i = 0; i < 8; i++) {
            if (((((addr1 >> (j * 8)) & 0xFF) & ( 1 << mask )) >> mask == 1) == ((((addr2 >> (j * 8)) & 0xFF) & ( 1 << mask )) >> mask == 1)) {
                len_match++;
            } else {
                return len_match;
            }
            mask--;
        }
    }
    return len_match;
}

__global__ void kernel_ip_lookup_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, RouteGPU *ip_list, uint32_t len) {
    int pkt_id = threadIdx.x;
    uint32_t item_id = blockIdx.x;

    struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ipv4;
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

            ipv4 = (struct rte_ipv4_hdr *)((char *)eth + sizeof(struct rte_ether_hdr));


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

            uint8_t longest = 0;
            uint32_t gateway = 0;

            for (uint32_t i = 0; i < len; i++) {
                uint32_t dst_addr = ((uint32_t) ipv4->dst_addr) & ip_list[i].mask;

                uint8_t len1 = longest_match(dst_addr, ip_list[i].addr);
                if (len1 > longest) {
                    gateway = ip_list[i].gw;
                    longest = len1;
                }


            }
            // printf("Src ip: %d.%d.%d.%d\n",    (ipv4->dst_addr >> (0 * 8)) & 0xFF,   (ipv4->dst_addr >> (1 * 8)) & 0xFF,   (ipv4->dst_addr >> (2 * 8)) & 0xFF,   (ipv4->dst_addr >> (3 * 8)) & 0xFF);
            // printf("gw: %d\n", gateway);
            // comm_list[item_id].pkt_list[pkt_id].size = gateway;
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

void wrapper_ip_lookup_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len) {
    kernel_ip_lookup_persistent <<< cuda_blocks, cuda_threads, 0, stream >>> (comm_list, comm_list_size, ip_list, len);
}

#endif /* HAVE_CUDA */