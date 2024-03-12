#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include "gpuethermirror.hh"
#include <click/cuda/cuda_utils.hh>

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// todo handle multithreading

int GPUEtherMirror::initialize(ErrorHandler *errh) {
    _comm_list_curr_index = 0;
    _comm_list_size = MAX_BURSTS_X_QUEUE;
    _comm_list = rte_gpu_comm_create_list(0, _comm_list_size);   // todo generify GPU
    if (_comm_list == NULL)  {
        return errh->error("Unable to create the GPU communication list");
    }

    wrapper_ether_mirror(_comm_list, _comm_list_size, 1, 256);  // todo generify cuda blocks & threads
    CUDA_CHECK(cudaGetLastError(), errh);

    return 0;
}

#if HAVE_BATCH
PacketBatch *
GPUEtherMirror::simple_action_batch(PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    enum rte_gpu_comm_list_status status;

    unsigned n = batch->count();
    rte_mbuf *array[n];

    int i = 0;
    FOR_EACH_PACKET(batch, p)
        array[i++] = p->uniqueify()->mb();

    rte_gpu_comm_get_status(&_comm_list[_comm_list_curr_index], &status);
    if (status != RTE_GPU_COMM_LIST_FREE) {
        errh->error("List is not free!\n");
        return batch;
    }
    
    int ret = rte_gpu_comm_populate_list_pkts(&_comm_list[_comm_list_curr_index], array, n);
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d\n", ret);
        return batch;
    }
    rte_wmb();

    do {
        rte_gpu_comm_get_status(&_comm_list[_comm_list_curr_index], &status);
    } while (status != RTE_GPU_COMM_LIST_DONE);

    rte_gpu_comm_cleanup_list(&_comm_list[_comm_list_curr_index]);

    _comm_list_curr_index = (_comm_list_curr_index + 1) % _comm_list_size;
    return batch;
}
#endif

void GPUEtherMirror::cleanup(CleanupStage) {
    rte_gpu_comm_destroy_list(_comm_list, _comm_list_size);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirror)