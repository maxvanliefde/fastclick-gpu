#include <click/config.h>
#include <click/cuda/cuda_crc32.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <click/crc32.h>
#include <rte_gpudev.h>

#include "gpusetcrc32commlist.hh"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

CLICK_DECLS

GPUSetCRC32CommList::GPUSetCRC32CommList() {};

int GPUSetCRC32CommList::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .complete() < 0)
        return -1;

    return 0;
}

int GPUSetCRC32CommList::initialize(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();

    /* Generate table and copy it on GPU */
    gen_crc_table(_crc_table);
    CUDA_CHECK(cudaMalloc(&_gpu_table, 256 * sizeof(uint32_t)), errh);
    CUDA_CHECK(cudaMemcpy(_gpu_table, _crc_table, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice), errh);

    /* Initialize n threads */
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
                wrapper_setcrc32_persistent(s.comm_lists[list_id].comm_list, s.comm_lists[list_id].comm_list_size, _gpu_table, _blocks_per_q, _max_batch, s.comm_lists[list_id].cuda_stream);
                CUDA_CHECK(cudaGetLastError(), errh);
            }
        }
    }

    if (_verbose) {
        click_chatter("%p{element}: %d usable threads, %d queue per thread, %d batch per queue.",
            this,
            _usable_threads.weight(), 1, _capacity
        );
        for (int th_id = 0; th_id < master()->nthreads(); th_id++)
            if (_usable_threads[th_id])
                click_chatter("%p{element}: Pipeline in thread %d", this, th_id);
    }

    return 0;
}

/* Sends the batch to the GPU */
void GPUSetCRC32CommList::push_batch(int port, PacketBatch *batch) {
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
    FOR_EACH_PACKET_SAFE(batch, p) {
        WritablePacket *q = p->put(4);
        array[i++] = reinterpret_cast<struct rte_mbuf *>(q);
    }
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
        wrapper_setcrc32_commlist(&list_state->comm_list[list_state->comm_list_put_index], _gpu_table, _blocks_per_q, _max_batch, list_state->cuda_stream);
    }

    list_state->comm_list_put_index = (list_state->comm_list_put_index + 1) % list_state->comm_list_size;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}


void GPUSetCRC32CommList::cleanup(CleanupStage cs) {
    cleanup_base(cs);
    cudaFree(_gpu_table);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUSetCRC32CommList)