#include <stdio.h>

#include <click/cuda/cuda_iplookup.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__device__ bool matches_prefix_coal(uint32_t addr1, uint32_t addr2, uint32_t mask)
{
    return ((addr1 ^ addr2) & mask) == 0;
}

__device__ uint32_t lookup_entry_coal(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    uint64_t found = 0;
    for (int i = 0; i < len; i++) {
        RouteGPU r = ip_list[i];
        bool b = matches_prefix_coal(a, r.addr, r.mask);
        if (b) found = i;
        
	}
    return found;
}

__device__ uint32_t lookup_route_coal(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    int ei = lookup_entry_coal(a, ip_list, len);
    
    if (ei >= 0) {
	return ip_list[ei].port;
    } else
	return -1;
}

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
    int index = 0;

    

    // printf("ip_list[0].port: %d\n", ip_list[0].port);

    /* Workload */
    if (pkt_id < n_pkts) {
        char *data = batch_memory + stride * pkt_id;
        src = (uint8_t *) data;


        uint32_t *dst_addr = (uint32_t *) (data);

        // printf("addr: %d\n", *dst_addr);
        // printf("IP address: %d.%d.%d.%d\n", (*dst_addr >> (0 * 8)) & 0xFF, (*dst_addr >> (1 * 8)) & 0xFF, (*dst_addr >> (2 * 8)) & 0xFF, (*dst_addr >> (3 * 8)) & 0xFF);
        uint32_t port = lookup_route_coal((uint32_t) *dst_addr, ip_list, len);
        src[0] = port;
    }

    __syncthreads();
}

void wrapper_iplookup(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len) {
    kernel_iplookup <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts, ip_list, len);
}


#endif /* HAVE_CUDA */