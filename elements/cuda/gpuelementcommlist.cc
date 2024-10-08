#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuelementcommlist.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUElementCommList::GPUElementCommList() : _state(), _capacity(4096), _max_batch(1024), _blocks_per_q(1), _block(false), _verbose(false), _lists_per_core(1), _persistent_kernel(true) {};

int GPUElementCommList::configure_base(Vector<String> &conf, ErrorHandler *errh) {
    if (Args(conf, this, errh)
        .bind(conf)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("THREAD_BLOCKS_PER_QUEUE", _blocks_per_q)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .read("LISTS_PER_CORE", _lists_per_core)
        .read_p("PERSISTENT_KERNEL", _persistent_kernel)
        .consume() < 0) 
    {
        return -1;
    }

    if (_max_batch > RTE_GPU_COMM_LIST_PKTS_MAX) {
        errh->error("Given MAX_BATCH = %d but the maximum is %d", _max_batch, RTE_GPU_COMM_LIST_PKTS_MAX);
        return -1;
    }

    if (_lists_per_core == 0) {
        errh->error("Given LISTS_PER_CORE = 0 but it has to be at least 1");
        return -1;
    }

    return 0;
}

int GPUElementCommList::initialize_base(ErrorHandler *errh, base_wrapper_persistent_kernel launch_kernel) {
    _usable_threads = get_pushing_threads();

    for (int th_id = 0; th_id < master()->nthreads(); th_id++) {
        if (!_usable_threads[th_id])
            continue;

        state &s = _state.get_value(th_id);

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);

        s.timer = new Timer(this);
        s.timer->initialize(this);
        s.timer->move_thread(th_id);
        s.timer->schedule_now();

        s.next_list_get = 0;
        s.next_list_put = 0;
        s.comm_lists = new comm_list_state[_lists_per_core];
        for (uint8_t list_id = 0; list_id < _lists_per_core; list_id++) {
            s.comm_lists[list_id].comm_list_put_index = 0;
            s.comm_lists[list_id].comm_list_get_index = 0;
            s.comm_lists[list_id].comm_list_size = _capacity;        
            s.comm_lists[list_id].comm_list = rte_gpu_comm_create_list(0, s.comm_lists[list_id].comm_list_size);   // todo generify GPU
            if (s.comm_lists[list_id].comm_list == NULL)  {
                return errh->error("Unable to create the GPU communication list for thread %d", th_id);
            }

            cudaError_t cuda_ret = cudaStreamCreate(&s.comm_lists[list_id].cuda_stream);
            CUDA_CHECK(cuda_ret, errh);
            if (_persistent_kernel) {
                launch_kernel(s.comm_lists[list_id].comm_list, s.comm_lists[list_id].comm_list_size, _blocks_per_q, _max_batch, s.comm_lists[list_id].cuda_stream);
                CUDA_CHECK(cudaGetLastError(), errh);
            }
        }
    }

    if (_verbose) {
        click_chatter("%p{element}: %d usable threads, %d list%s per thread, %d batch per list.",
            this,
            _usable_threads.weight(), _lists_per_core, _lists_per_core == 1 ? "" : "s", _capacity
        );
        for (int th_id = 0; th_id < master()->nthreads(); th_id++)
            if (_usable_threads[th_id])
                click_chatter("%p{element}: Pipeline in thread %d", this, th_id);
    }

    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUElementCommList::push_batch(int port, PacketBatch *batch) {
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

    // populate lists in round robin fashion
    struct comm_list_state *list_state = &_state->comm_lists[_state->next_list_put];
    _state->next_list_put = (_state->next_list_put + 1) % _lists_per_core;

    do {
        ret = rte_gpu_comm_get_status(&list_state->comm_list[list_state->comm_list_put_index], &status);
        if (ret != 0) {
            errh->error("rte_gpu_comm_get_status returned error %d\n", ret);
            batch->kill();
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
    
    ret = rte_gpu_comm_populate_list_pkts(&list_state->comm_list[list_state->comm_list_put_index], array, n);     // includes call to gpu_wmb
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d (size was %d )\n", ret, n);
        batch->kill();
        return;
    }

    if (likely(_persistent_kernel)) {
        rte_wmb();
    } else {
        wrapper_ether_mirror(&list_state->comm_list[list_state->comm_list_put_index], _max_batch);
    }

    list_state->comm_list_put_index = (list_state->comm_list_put_index + 1) % list_state->comm_list_size;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

bool GPUElementCommList::run_task(Task *task) {
    enum rte_gpu_comm_list_status status;
    uint32_t n;
    struct rte_mbuf **pkts;

    // read lists in round robin fashion
    struct comm_list_state *list_state = &_state->comm_lists[_state->next_list_get];
    _state->next_list_get = (_state->next_list_get + 1) % _lists_per_core;

    rte_gpu_comm_get_status(&list_state->comm_list[list_state->comm_list_get_index], &status);
    if (status != RTE_GPU_COMM_LIST_DONE) {
        task->fast_reschedule();
        return false;
    }

    rte_rmb();

    /* Batch the processed packets */
    PacketBatch *head = nullptr;
    WritablePacket *packet, *last;
    n = list_state->comm_list[list_state->comm_list_get_index].num_pkts;
    pkts = list_state->comm_list[list_state->comm_list_get_index].mbufs;
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

    rte_gpu_comm_cleanup_list(&list_state->comm_list[list_state->comm_list_get_index]);
    rte_mb();

    list_state->comm_list_get_index = (list_state->comm_list_get_index + 1) % list_state->comm_list_size;
    return true;
}

void GPUElementCommList::run_timer(Timer *timer) {
    _state->task->reschedule();
    timer->schedule_now();
}

#endif

bool GPUElementCommList::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    return true;
}


/* Cleans all structures */
void GPUElementCommList::cleanup_base(CleanupStage) {
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
        for (uint8_t list_id = 0; list_id < _lists_per_core; list_id++) {
            for (int index = 0; index < s.comm_lists[list_id].comm_list_size; index++) {
                ret = rte_gpu_comm_set_status(&s.comm_lists[list_id].comm_list[index], RTE_GPU_COMM_LIST_ERROR);
                if (ret != 0) {
                    errh->error(
                        "Can't set status RTE_GPU_COMM_LIST_ERROR on item %d for thread i."
                        "This probably means that the GPU kernel associated will never finish!", index, i);
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(s.comm_lists[list_id].cuda_stream), errh);

            /* Destroy the stream and the communication list */
            cudaStreamDestroy(s.comm_lists[list_id].cuda_stream);
            // rte_gpu_comm_destroy_list(s.comm_lists[list_id].comm_list, s.comm_lists[list_id].comm_list_size);    // does not work
        }
        delete[] s.comm_lists;
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUElementCommList)