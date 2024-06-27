#include <stdio.h>

#include <click/cuda/cuda_ethermirror.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__global__ void kernel_ether_mirror(char *batch_memory, uint32_t n_pkts) {
    int pkt_id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("griddim %d blockdim %d threadidx %d", gridDim.x, blockDim.x, threadIdx.x);
    uint8_t *src, *dst, tmp[RTE_ETHER_ADDR_LEN];
    // uint8_t src[RTE_ETHER_ADDR_LEN], dst[RTE_ETHER_ADDR_LEN], tmp[RTE_ETHER_ADDR_LEN];
    int stride = 2*RTE_ETHER_ADDR_LEN;

    /* Workload */
    if (pkt_id < n_pkts) {
        char *data = batch_memory + stride * pkt_id;
        src = (uint8_t *) data;
        dst = (uint8_t *) data + RTE_ETHER_ADDR_LEN;
        // printf("cuda: writing in %p\n", data);

        /* Verify source and dest of ethernet addresses */
        // printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);

        /* Swap addresses */
        uint8_t j;
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) tmp[j] = src[j];
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) src[j] = dst[j];
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) dst[j] = tmp[j];

        /* Verify source and dest of ethernet addresses */
        // printf("After Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
    }

    __syncthreads();
}

void wrapper_ether_mirror(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream) {
    kernel_ether_mirror <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts);
}


__global__ void kernel_ether_mirror_graph(char *batch_memory, uint32_t *n_pkts) {
    int pkt_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *src, *dst, tmp[RTE_ETHER_ADDR_LEN];
    int stride = 2*RTE_ETHER_ADDR_LEN;

    /* Workload */
    if (pkt_id < *n_pkts) {
        char *data = batch_memory + stride * pkt_id;
        src = (uint8_t *) data;
        dst = (uint8_t *) data + RTE_ETHER_ADDR_LEN;

        /* Verify source and dest of ethernet addresses */
        // printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);

        /* Swap addresses */
        uint8_t j;
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) tmp[j] = src[j];
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) src[j] = dst[j];
        for (j = 0; j < RTE_ETHER_ADDR_LEN; j++) dst[j] = tmp[j];

        /* Verify source and dest of ethernet addresses */
        // printf("After Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
    }

    __syncthreads();
}

void wrapper_ether_mirror_graph(char *batch_memory, uint32_t *n_pkts, int blocks, int threads, cudaStream_t stream) {
    kernel_ether_mirror_graph <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts);
}


#endif /* HAVE_CUDA */