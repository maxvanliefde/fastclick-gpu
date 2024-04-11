#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuethermirror.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUEtherMirror::GPUEtherMirror() : _state(), _capacity(4096), _max_batch(1024), _blocks_per_q(1), _block(false), _verbose(false) {};

int GPUEtherMirror::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (Args(conf, this, errh)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("THREAD_BLOCKS_PER_QUEUE", _blocks_per_q)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .complete() < 0) 
    {
        return -1;
    }

    if (_max_batch > RTE_GPU_COMM_LIST_PKTS_MAX) {
        errh->error("Given MAX_BATCH = %d but the maximum is %d", _max_batch, RTE_GPU_COMM_LIST_PKTS_MAX);
        return -1;
    }

    return 0;
}

int GPUEtherMirror::initialize(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();

    for (int th_id = 0; th_id < master()->nthreads(); th_id++) {
        if (!_usable_threads[th_id])
            continue;

        state &s = _state.get_value(th_id);
        s.comm_list_put_index = 0;
        s.comm_list_get_index = 0;
        s.comm_list_size = _capacity;
        s.comm_list = rte_gpu_comm_create_list(0, s.comm_list_size);   // todo generify GPU
        if (s.comm_list == NULL)  {
            return errh->error("Unable to create the GPU communication list for thread %d", th_id);
        }

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);

        s.timer = new Timer(this);
        s.timer->initialize(this);
        s.timer->move_thread(th_id);
        s.timer->schedule_now();
        
        cudaError_t cuda_ret = cudaStreamCreate(&s.cuda_stream);
        CUDA_CHECK(cuda_ret, errh);
        wrapper_ether_mirror_persistent(s.comm_list, s.comm_list_size, _blocks_per_q, _max_batch, s.cuda_stream);
        CUDA_CHECK(cudaGetLastError(), errh);
    }

    if (_verbose) {
        errh->message("%s{element}: %d usable threads, %d queue per thread, %d batch per queue."
            "Be sure your mempool can hold at least %d batches, unless you may suffer a loss of performance",
            this,
            _usable_threads.weight(), 1, _capacity,
            _usable_threads.weight() * 1 * _capacity
        );
    }


    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUEtherMirror::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    enum rte_gpu_comm_list_status status;
    int ret;

    unsigned n = batch->count();

    if (unlikely(n > _max_batch)) {
        PacketBatch* todrop;
        batch->split(_max_batch, todrop, true);
        todrop->kill();
        if (unlikely(_verbose)) 
            click_chatter("%p{element} Warning: a batch of size %d has been given, but the max size is %d. "
            "Dropped %d packets", this, n, _max_batch, n-_max_batch);
        n = batch->count();
    }

    rte_mbuf *array[n];

    int i = 0;
    FOR_EACH_PACKET(batch, p)
        array[i++] = reinterpret_cast<struct rte_mbuf *>(p);
    
    do {
        ret = rte_gpu_comm_get_status(&_state->comm_list[_state->comm_list_put_index], &status);
        if (ret != 0) {
            errh->error("rte_gpu_comm_get_status returned error %d\n", ret);
            return;
        }

        if (status != RTE_GPU_COMM_LIST_FREE) {
            if (_block) {
                _state->task->reschedule();
                if (unlikely(_verbose))
                    click_chatter("%p{element} Congestion: List is not free!" 
                        "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this);
            } else {
                batch->kill();
                if (unlikely(_verbose))
                    click_chatter("%p{element} Dropped %d packets: List is not free!" 
                        "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this, n);

                return;
            }
        }
    } while(unlikely(status != RTE_GPU_COMM_LIST_FREE));
    
    ret = rte_gpu_comm_populate_list_pkts(&_state->comm_list[_state->comm_list_put_index], array, n);     // includes call to gpu_wmb
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d (size was %d )\n", ret, n);
        batch->kill();
        return;
    }
    rte_wmb();

    _state->comm_list_put_index = (_state->comm_list_put_index + 1) % _state->comm_list_size;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

bool GPUEtherMirror::run_task(Task *task) {
    enum rte_gpu_comm_list_status status;
    uint32_t n;
    struct rte_mbuf **pkts;

    rte_gpu_comm_get_status(&_state->comm_list[_state->comm_list_get_index], &status);
    if (status != RTE_GPU_COMM_LIST_DONE) {
        task->fast_reschedule();
        return false;
    }

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

void GPUEtherMirror::run_timer(Timer *timer) {
    _state->task->reschedule();
    timer->schedule_now();
}

#endif

bool GPUEtherMirror::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    return true;
}


/* Cleans all structures */
void GPUEtherMirror::cleanup(CleanupStage) {
    ErrorHandler *errh = ErrorHandler::default_handler();

    for (int i = 0; i < _state.weight(); i++) {
        if (!_usable_threads[i])
            continue;

        state &s = _state.get_value(i);

        /* Finish all the tasks polling on the GPU communication list */
        delete s.task;
        s.task = nullptr;

        /* Terminating CUDA kernels */
        int ret;
        for (int index = 0; index < s.comm_list_size; index++) {
            ret = rte_gpu_comm_set_status(&s.comm_list[index], RTE_GPU_COMM_LIST_ERROR);
            if (ret != 0) {
                errh->error(
                    "Can't set status RTE_GPU_COMM_LIST_ERROR on item %d for thread i."
                    "This probably means that the GPU kernel associated will never finish!", index, i);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(s.cuda_stream), errh);

        /* Destroy the stream and the communication list */
        cudaStreamDestroy(s.cuda_stream);
        // rte_gpu_comm_destroy_list(s.comm_list, s.comm_list_size);    // does not work
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirror)