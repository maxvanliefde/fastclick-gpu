#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>

#include "gpuethermirror.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUEtherMirror::GPUEtherMirror() : _state() {};

int GPUEtherMirror::initialize(ErrorHandler *errh) {
    printf("Initializing with %d threads\n", master()->nthreads());
    for (int th_id = 0; th_id < master()->nthreads(); th_id++) {
        state &s = _state.get_value(th_id);
        s.comm_list_put_index = 0;
        s.comm_list_get_index = 0;
        s.comm_list_size = MAX_BURSTS_X_QUEUE;
        s.comm_list = rte_gpu_comm_create_list(0, s.comm_list_size);   // todo generify GPU
        if (s.comm_list == NULL)  {
            return errh->error("Unable to create the GPU communication list for thread %d", th_id);
        }

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);
        
        cudaError_t cuda_ret = cudaStreamCreate(&s.cuda_stream);
        CUDA_CHECK(cuda_ret, errh);
        wrapper_ether_mirror(s.comm_list, s.comm_list_size, 1, 1024, s.cuda_stream);  // todo generify cuda blocks & threads
        CUDA_CHECK(cudaGetLastError(), errh);
    }

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
        array[i++] = p->mb();
    
    int ret = rte_gpu_comm_get_status(&_state->comm_list[_state->comm_list_put_index], &status);
    if (ret != 0) {
        errh->error("rte_gpu_comm_get_status returned error %d\n", ret);
        return;
    }

    if (status != RTE_GPU_COMM_LIST_FREE) {
        errh->error("List is not free! GPU processing is likely too slow\n");
        return;
    }
    
    ret = rte_gpu_comm_populate_list_pkts(&_state->comm_list[_state->comm_list_put_index], array, n);     // includes call to gpu_wmb
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d (size was %d )\n", ret, n);
        return;
    }
    rte_wmb();

    _state->comm_list_put_index = (_state->comm_list_put_index + 1) % _state->comm_list_size;
    _state->task->reschedule();
}

bool GPUEtherMirror::run_task(Task *task) {
    enum rte_gpu_comm_list_status status;
    uint32_t n;
    struct rte_mbuf **pkts;

    do {
        rte_gpu_comm_get_status(&_state->comm_list[_state->comm_list_get_index], &status);
    } while (status != RTE_GPU_COMM_LIST_DONE);

    rte_rmb();

    /* Batch the processed packets */
    PacketBatch *head = nullptr;
    WritablePacket *packet, *last;
    n = _state->comm_list[_state->comm_list_get_index].num_pkts;
    pkts = _state->comm_list[_state->comm_list_get_index].mbufs;
    for (uint32_t i = 0; i < n; i++) {
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

    rte_gpu_comm_cleanup_list(&_state->comm_list[_state->comm_list_get_index]);
    rte_mb();
    _state->comm_list_get_index = (_state->comm_list_get_index + 1) % _state->comm_list_size;
    return true;
}

#endif

// todo fix, close cuda kernels and destroy streams
void GPUEtherMirror::cleanup(CleanupStage) {
    rte_gpu_comm_destroy_list(_state->comm_list, _state->comm_list_size);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirror)