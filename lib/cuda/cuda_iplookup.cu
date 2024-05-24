#include <stdio.h>

#include <click/cuda/cuda_iplookup.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__device__ uint8_t longest_match2(uint32_t addr1, uint32_t addr2) {
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

__global__ void kernel_iplookup(char *batch_memory, uint32_t n_pkts, RouteGPU *ip_list, uint32_t len) {
    int pkt_id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("griddim %d blockdim %d threadidx %d", gridDim.x, blockDim.x, threadIdx.x);
    uint8_t *src, *dst, tmp[6];
    int stride = 8;

    // printf("hello les bebz\n");
    // printf("ip_list[0].addr: %d\n", ip_list[0].addr);

    /* Workload */
    if (pkt_id < n_pkts) {
        char *data = batch_memory + stride * pkt_id;
        // src = (uint8_t *) data;
        // dst = (uint8_t *) data + 6;
        // uint8_t* dst2 = (uint8_t *) data + 4;
        // printf("src: %d\n", src);
        // printf("dst: %d\n", dst);

        /* Verify source and dest of ethernet addresses */
        // printf("Before Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);

        /* Verify source and dest of internet addresses */
        // printf("Before Swap, Source: %02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], 
        //     dst2[0], dst2[1], dst2[2], dst2[3]);

        // uint32_t *src_addr = (uint32_t *) (data);
        uint32_t *dst_addr = (uint32_t *) (data);

        uint8_t longest = 0;
        uint32_t gateway = 0;

        for (uint32_t i = 0; i < len; i++) {
            uint32_t input = (*dst_addr) & ip_list[i].mask;
            // printf("idk man %d\n", input);

            uint8_t len1 = longest_match2(input, ip_list[i].addr);
            if (len1 > longest) {
                gateway = ip_list[i].gw;
                longest = len1;
            }


        }

        // printf("aaaaaaaaah %d\n", longest);

        // printf("Before Swap, Source: %02x\n",
        //     (uint32_t) src);

        /* Swap addresses */
        // uint8_t j;
        // for (j = 0; j < 6; j++) tmp[j] = src[j];
        // for (j = 0; j < 6; j++) src[j] = dst[j];
        // for (j = 0; j < 6; j++) dst[j] = tmp[j];

        /* Verify source and dest of ethernet addresses */
        // printf("After Swap, Source: %02x:%02x:%02x:%02x:%02x:%02x Dest: %02x:%02x:%02x:%02x:%02x:%02x\n",
        //     src[0], src[1], src[2], src[3], src[4], src[5], 
        //     dst[0], dst[1], dst[2], dst[3], dst[4], dst[5]);
    }

    __syncthreads();
}

void wrapper_iplookup(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len) {
    kernel_iplookup <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts, ip_list, len);
}


#endif /* HAVE_CUDA */