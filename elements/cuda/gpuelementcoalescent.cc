#include <click/config.h>
#include <click/cuda/cuda_ethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <click/algorithm.hh>

#include "gpuelementcoalescent.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUElementCoalescent::GPUElementCoalescent() : _state(), _capacity(4096), _max_batch(1024), _block(false), _verbose(false), _zc(true) {};

int GPUElementCoalescent::configure_base(Vector<String> &conf, ErrorHandler *errh) {
    if (Args(conf, this, errh)
        .bind(conf)
        .read_p("FROM", _from)
        .read_p("TO", _to)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .read_p("ZEROCOPY", _zc)
        .consume() < 0) 
    {
        return -1;
    }

    /* Compute stride */
    if (_to <= _from) {
        return errh->error("TO should be greather than FROM, but from = %d and to = %d", _to, _from);
    }
    _stride = _to - _from;

    /* Check if power of two and compute logarithm */
    uint32_t n = _max_batch;
    if (!is_pow2(n))
        return errh->error("MAX_BATCH should be a power of two, but %d was given", n);
    _log_max_batch = __builtin_ctz(n);

    /* Compute number of blocks */
    if (_max_batch >= 128) {
        _cuda_blocks = _max_batch / 128;
        _cuda_threads = 128;
    } else {
        _cuda_blocks = 1;
        _cuda_threads = _max_batch;
    }

    return 0;
}

int GPUElementCoalescent::initialize_base(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();

    size_t size_per_thread = ((size_t) _stride *  _capacity) << _log_max_batch;

    if (_verbose) {
        const char *str = "%p{element} Initialization:\n"
        "* from = %d, to = %d => %d bytes will be read per packet\n"
        "* a batch is %d maximum packets\n"
        "* %ld B will be allocated per thread\n";
        click_chatter(str, this, _from, _to, _stride, _max_batch, size_per_thread);
    }

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

        cudaError_t cuda_ret = cudaStreamCreate(&s.cuda_stream);
        CUDA_CHECK_RET(cuda_ret, errh);

        if (_zc) {
            // Under Unified Virtual Addressing, host and device pointers are the same. 
            cuda_ret = cudaHostAlloc((void **) &s.h_memory, size_per_thread, cudaHostAllocMapped);
            CUDA_CHECK_RET(cuda_ret, errh);
            s.d_memory = s.h_memory;
        } else {
            cuda_ret = cudaHostAlloc(&s.h_memory, size_per_thread, cudaHostAllocDefault);
            CUDA_CHECK_RET(cuda_ret, errh);
            cuda_ret = cudaMalloc((void **) &s.d_memory, size_per_thread);
            CUDA_CHECK_RET(cuda_ret, errh);
        }

        s.put_index = 0;
        s.get_index = 0;
        s.events = new cudaEvent_t[_capacity];
        s.batches = new PacketBatch*[_capacity];
        for (int i = 0; i < _capacity; i++) {
            cudaEventCreateWithFlags(&s.events[i], cudaEventDisableTiming);
            s.batches[i] = nullptr;
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

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUElementCoalescent::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    cudaError_t ret;

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

    char *h_batch_memory = _state->h_memory + (_state->put_index << _log_max_batch);
    char *loop_ptr = h_batch_memory;

    FOR_EACH_PACKET(batch, p) {
        const unsigned char *data = p->data();
        memcpy(loop_ptr, data + _from, _stride);
        loop_ptr += _stride;
    }

    while (unlikely(_state->batches[_state->put_index] != nullptr)) {
        if (_block) {
            // TODO fix blocking mode, it never switches to the task 
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

    _state->batches[_state->put_index] = batch;

    if (_zc) {
        wrapper_ether_mirror(h_batch_memory, n, _cuda_blocks, _cuda_threads, _state->cuda_stream);
        cudaEventRecord(_state->events[_state->put_index], _state->cuda_stream);
    } else {
        char *d_batch_memory = _state->d_memory + (_state->put_index << _log_max_batch);
        size_t size = loop_ptr - h_batch_memory;
        cudaMemcpyAsync(d_batch_memory, h_batch_memory, size, cudaMemcpyHostToDevice, _state->cuda_stream);
        wrapper_ether_mirror(d_batch_memory, n, _cuda_blocks, _cuda_threads, _state->cuda_stream);
        cudaMemcpyAsync(h_batch_memory, d_batch_memory, size, cudaMemcpyDeviceToHost, _state->cuda_stream);
        cudaEventRecord(_state->events[_state->put_index], _state->cuda_stream);
    }
    
    _state->put_index = (_state->put_index + 1) % _capacity;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

/* Gets the computed batch from the GPU */
bool GPUElementCoalescent::run_task(Task *task) {
    uint32_t n;
    struct rte_mbuf **pkts;
    cudaError_t status;

    if (_state->batches[_state->get_index] == nullptr) {
        task->fast_reschedule();
        return false;
    }

    status = cudaEventQuery(_state->events[_state->get_index]);
    if (status != cudaSuccess) {
        task->fast_reschedule();
        return false;
    }

    /* Copy back to the packet */
    char *h_batch_memory = _state->h_memory + (_state->get_index << _log_max_batch);
    char *loop_ptr = h_batch_memory;
    PacketBatch *batch = _state->batches[_state->get_index];
    FOR_EACH_PACKET_SAFE(batch, p) {
        WritablePacket *q = p->uniqueify();
        unsigned char *data = q->data();
        memcpy(data + _from, loop_ptr, _stride);
        p = q;
    }

    output_push_batch(0, batch);
    _state->batches[_state->get_index] = nullptr;
    _state->get_index = (_state->get_index + 1) % _capacity;
    return true;
}

void GPUElementCoalescent::run_timer(Timer *timer) {
    _state->task->reschedule();
    timer->schedule_now();
}

#endif

bool GPUElementCoalescent::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    return true;
}


/* Cleans all structures */
void GPUElementCoalescent::cleanup_base(CleanupStage) {
    ErrorHandler *errh = ErrorHandler::default_handler();

    for (int i = 0; i < _state.weight(); i++) {
        if (!_usable_threads[i])
            continue;

        state &s = _state.get_value(i);

        /* Finish all the tasks polling on the GPU communication list */
        delete s.task;
        s.task = nullptr;

        /* Destroy CUDA resources */
        cudaStreamDestroy(s.cuda_stream);
        cudaFreeHost(&s.h_memory);
        for (int i = 0; i < _capacity; i++) {
            cudaEventDestroy(s.events[i]);
        }
        delete[] s.events;
        delete[] s.batches;
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUElementCoalescent)