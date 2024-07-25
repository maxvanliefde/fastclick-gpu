#include <stdio.h>

#include <click/cuda/cuda_iplookup.hh>
#include <rte_ether.h>

#ifdef HAVE_CUDA

__device__ bool matches_prefix_coal(uint32_t addr1, uint32_t addr2, uint32_t mask)
{
    return ((addr1 ^ addr2) & mask) == 0;
}

__device__ bool mask_as_specific_coal(uint32_t mask, uint32_t addr) {
    return (addr & mask) == mask;
}

__device__ int lookup_entry_coal(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    int found = -1;
    for (int i = 0; i < len - 1; i++) {
        RouteGPU r = ip_list[i];
	    bool b = matches_prefix_coal(a, r.addr, r.mask);
        if (b) {
            found = i;
            for (int j = r.extra; j < len - 1; j++) {
                RouteGPU s = ip_list[j];
                bool c = (matches_prefix_coal(a, s.addr, s.mask) && mask_as_specific_coal(r.mask, s.mask));
                if (c) found = j;
            }
        }

	}
    return found;
}

__device__ uint32_t lookup_route_coal(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    int ei = lookup_entry_coal(a, ip_list, len);
    
    // printf("port: %d\n", ei);
    
    if (ei >= 0) {
	// return ip_list[ei].port;
	return ei;
    } else
	return 0;
}

__global__ void kernel_iplookup(char *batch_memory, uint32_t n_pkts, RouteGPU *ip_list, uint32_t len) {    
    int pkt_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t *src;
    int stride = 4;

    /* Workload */
    if (pkt_id < n_pkts) {
        char *data = batch_memory + stride * pkt_id;
        src = (uint8_t *) data;

        uint32_t *dst_addr = (uint32_t *) (data);

        uint32_t port = lookup_route_coal((uint32_t) *dst_addr, ip_list, len);
        src[0] = port;
    }

    __syncthreads();
}

void wrapper_iplookup(char *batch_memory, uint32_t n_pkts, int blocks, int threads, cudaStream_t stream, RouteGPU *ip_list, uint32_t len) {
    kernel_iplookup <<< blocks, threads, 0, stream >>> (batch_memory, n_pkts, ip_list, len);
}


#endif /* HAVE_CUDA */