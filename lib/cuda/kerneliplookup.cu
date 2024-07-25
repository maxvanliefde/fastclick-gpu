#include <stdio.h>

#include <click/cuda/kerneliplookup.hh>
#include <rte_ether.h>
#include <rte_ip.h>

#ifdef HAVE_CUDA

__device__ bool matches_prefix(uint32_t addr1, uint32_t addr2, uint32_t mask)
{
    return ((addr1 ^ addr2) & mask) == 0;
}

__device__ bool mask_as_specific(uint32_t mask, uint32_t addr) {
    return (addr & mask) == mask;
}

__device__ uint32_t lookup_entry(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    uint64_t found = -1;
    for (int i = 0; i < len - 1; i++) {
        RouteGPU r = ip_list[i];
	    bool b = matches_prefix(a, r.addr, r.mask);
        if (b) {
            found = i;
            for (int j = r.extra; j < len - 1; j++) {
                RouteGPU s = ip_list[j];
                bool c = (matches_prefix(a, s.addr, s.mask) && mask_as_specific(r.mask, s.mask));
                if (c) found = j;
            }
        }

	}
    return found;
}

__device__ uint32_t lookup_route(uint32_t a, RouteGPU *ip_list, uint32_t len) 
{
    int ei = lookup_entry(a, ip_list, len);

    if (ei >= 0) {
	return ip_list[ei].port;
    } else
	return 0;
}

__global__ void kernel_ip_lookup(struct rte_gpu_comm_list *comm_list_item, RouteGPU *ip_list, uint32_t len) {
    int pkt_id = threadIdx.x;

    struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ipv4;

    uint8_t *src, *dst, tmp[6];
    // size_t size;

    /* Workload */
    if (pkt_id < comm_list_item[0].num_pkts && comm_list_item[0].pkt_list[pkt_id].addr != NULL) {
        eth = (struct rte_ether_hdr *)(((uint8_t *) (comm_list_item[0].pkt_list[pkt_id].addr)));
        ipv4 = (struct rte_ipv4_hdr *)((char *)eth + sizeof(struct rte_ether_hdr));

        // printf("total size: %d\n", ipv4->total_length);

        // find the route in the table
        uint32_t port = lookup_route((uint32_t) ipv4->dst_addr, ip_list, len);

        src = (uint8_t *) (&eth->src_addr);
        dst = (uint8_t *) (&eth->dst_addr);

        uint8_t j;
        for (j = 0; j < 6; j++) tmp[j] = src[j];
        for (j = 0; j < 6; j++) src[j] = dst[j];
        for (j = 0; j < 6; j++) dst[j] = tmp[j];

        // save the result on the commlist
        comm_list_item[0].pkt_list[pkt_id].size = port;
        
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

void wrapper_ip_lookup(struct rte_gpu_comm_list *comm_list_item, int cuda_threads, RouteGPU *ip_list, uint32_t len) {
    kernel_ip_lookup <<< 1, cuda_threads >>> (comm_list_item, ip_list, len);
}

__global__ void kernel_ip_lookup_persistent(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, RouteGPU *ip_list, uint32_t len) {
    int pkt_id = threadIdx.x;
    uint32_t item_id = blockIdx.x;

    struct rte_ether_hdr *eth;
    struct rte_ipv4_hdr *ipv4;

    uint8_t *src, *dst, tmp[6];
    // size_t size;

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
            ipv4 = (struct rte_ipv4_hdr *)((char *)eth + sizeof(struct rte_ether_hdr));

            // find the route in the table
            uint32_t port = lookup_route((uint32_t) ipv4->dst_addr, ip_list, len);

            // save the result of the iplookup on the commlist
            comm_list[item_id].pkt_list[pkt_id].size = port;

            /* EtherMirror */
            src = (uint8_t *) (&eth->src_addr);
            dst = (uint8_t *) (&eth->dst_addr);

            uint8_t j;
            for (j = 0; j < 6; j++) tmp[j] = src[j];
            for (j = 0; j < 6; j++) src[j] = dst[j];
            for (j = 0; j < 6; j++) dst[j] = tmp[j];
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