#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>

#include "gpuethermirror.hh"


CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// todo handle multithreading

int GPUEtherMirror::initialize(ErrorHandler *errh) {
    _comm_list_put_index = _comm_list_get_index = 0;
    _comm_list_size = MAX_BURSTS_X_QUEUE;
    _comm_list = rte_gpu_comm_create_list(0, _comm_list_size);   // todo generify GPU
    if (_comm_list == NULL)  {
        return errh->error("Unable to create the GPU communication list");
    }

    _task = new Task(this);
    ScheduleInfo::initialize_task(this, _task, false, errh);

    wrapper_ether_mirror(_comm_list, _comm_list_size, 1, 256);  // todo generify cuda blocks & threads
    CUDA_CHECK(cudaGetLastError(), errh);

    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUEtherMirror::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    enum rte_gpu_comm_list_status status;

    unsigned n = batch->count();
    rte_mbuf *array[n];

    int i = 0;
    FOR_EACH_PACKET(batch, p)
        array[i++] = p->uniqueify()->mb();

    rte_gpu_comm_get_status(&_comm_list[_comm_list_put_index], &status);
    if (status != RTE_GPU_COMM_LIST_FREE) {
        errh->error("List is not free! GPU processing is likely too slow\n");
        return;
    }
    
    int ret = rte_gpu_comm_populate_list_pkts(&_comm_list[_comm_list_put_index], array, n);     // includes call to gpu_wmb
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d\n", ret);
        return;
    }
    rte_wmb();

    _comm_list_put_index = (_comm_list_put_index + 1) % _comm_list_size;
    _task->reschedule();
}

bool GPUEtherMirror::run_task(Task *task) {
    enum rte_gpu_comm_list_status status;
    uint32_t n;
    struct rte_mbuf **pkts;

    do {
        rte_gpu_comm_get_status(&_comm_list[_comm_list_get_index], &status);
    } while (status != RTE_GPU_COMM_LIST_DONE);

    rte_rmb();

    /* Batch the processed packets */
    PacketBatch *head = nullptr;
    WritablePacket *packet, *last;
    n = _comm_list[_comm_list_get_index].num_pkts;
    pkts = _comm_list[_comm_list_get_index].mbufs;
    for (uint32_t i = 0; i < n; i++) {
        unsigned char *data = rte_pktmbuf_mtod(pkts[i], unsigned char *);
        packet = static_cast<WritablePacket *>(Packet::make(pkts[i]));
        if (head == NULL) 
            head = PacketBatch::start_head(packet);
        else
            last->set_next(packet);
        last = packet;
    }
    if (head) {
        head->make_tail(last, n);
        output_push_batch(0, head);
    }

    rte_gpu_comm_cleanup_list(&_comm_list[_comm_list_get_index]);
    rte_mb();
    _comm_list_get_index = (_comm_list_get_index + 1) % _comm_list_size;
    return true;
}

/*
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
*/
#endif

void GPUEtherMirror::cleanup(CleanupStage) {
    rte_gpu_comm_destroy_list(_comm_list, _comm_list_size);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirror)