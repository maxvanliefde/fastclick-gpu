#include <click/config.h>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <click/cuda/cuda_iplookup.hh>

#include "gpuiplookupcoalescent.hh"

CLICK_DECLS

GPUIPLookupWithCopy::GPUIPLookupWithCopy() : _ip_list_len(1), _ip_list_cpu() {};
// GPUIPLookupWithCopy::GPUIPLookupWithCopy() : _ip_list_len(1), _ip_list_cpu(), _ip_list_gpu() {};

bool GPUIPLookupWithCopy::cp_ip_route(String s, Route *r_store, bool remove_route, Element *context)
{
    Route r;
    if (!IPPrefixArg(true).parse(cp_shift_spacevec(s), r.addr, r.mask, context))
	return false;
    r.addr &= r.mask;


    String word = cp_shift_spacevec(s);
    if (word == "-")
	/* null gateway; do nothing */;
    else if (IPAddressArg().parse(word, r.gw, context))
	/* do nothing */;
    else
	goto two_words;

    word = cp_shift_spacevec(s);
  two_words:
    if (IntArg().parse(word, r.port) || (!word && remove_route))
	if (!cp_shift_spacevec(s)) { // nothing left
	    *r_store = r;
	    return true;
	}

    return false;
}

void GPUIPLookupWithCopy::print_route(Route route) {
    printf("IP address: %d.%d.%d.%d\n", (route.addr >> (0 * 8)) & 0xFF, (route.addr >> (1 * 8)) & 0xFF, (route.addr >> (2 * 8)) & 0xFF, (route.addr >> (3 * 8)) & 0xFF);
    printf("Mask: %d.%d.%d.%d\n",       (route.mask >> (0 * 8)) & 0xFF, (route.mask >> (1 * 8)) & 0xFF, (route.mask >> (2 * 8)) & 0xFF, (route.mask >> (3 * 8)) & 0xFF);
    printf("Gateway: %d.%d.%d.%d\n",    (route.gw >> (0 * 8)) & 0xFF,   (route.gw >> (1 * 8)) & 0xFF,   (route.gw >> (2 * 8)) & 0xFF,   (route.gw >> (3 * 8)) & 0xFF);
    printf("Port: %d\n", route.port);
}

int GPUIPLookupWithCopy::configure(Vector<String> &conf, ErrorHandler *errh) {

    int ret = 0, r1, eexist = 0;
    struct Route route;

    _ip_list_cpu = (Route*) malloc(sizeof(Route) * conf.size());

    if (Args(conf, this, errh)
        .bind(conf)
        .read_mp("FROM", _from)
        .read_mp("TO", _to)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .read_p("ZEROCOPY", _zc)
        .read_p("QUEUES_PER_CORE", _queues_per_core)
        .consume() < 0) 
    {
        return -1;
    }

    /* Compute stride */
    if (_to <= _from) {
        return errh->error("TO should be greather than FROM, but from = %d and to = %d", _to, _from);
    }
    _stride = _to - _from;
    // if (_stride != 4) {
    //     return errh->error("TO - FROM should be 4, but from = %d and to = %d", _to, _from);
    // }

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

    // printf("conf.size() = %d\n", conf.size());

    for (int i = 0; i < conf.size(); i++) {
        if (!cp_ip_route(conf[i], &route, false, this)) {
            errh->error("argument %d should be %<ADDR/MASK [GATEWAY] OUTPUT%>", i+1);
            ret = -EINVAL;
        } else if (route.port < 0 || route.port >= noutputs()) {
            errh->error("argument %d bad OUTPUT", i+1);
            ret = -EINVAL;
        }

        // print_route(route);

        memcpy(&(_ip_list_cpu[i]), &route, sizeof(Route));
    }
    _ip_list_len = conf.size();
    if (eexist)
	errh->warning("%d %s replaced by later versions", eexist, eexist > 1 ? "routes" : "route");
    return ret;

    return 0;
}

int GPUIPLookupWithCopy::initialize(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();

    cudaMalloc(&_ip_list_gpu, _ip_list_len*sizeof(RouteGPU));
    cudaMemcpy(_ip_list_gpu, _ip_list_cpu, _ip_list_len*sizeof(RouteGPU), cudaMemcpyHostToDevice);

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

        s.queues = new struct queue_state[_queues_per_core];
        s.next_queue_put = 0;
        s.next_queue_get = 0;
        for (uint8_t queue_id = 0; queue_id < _queues_per_core; queue_id++) {
            struct queue_state *queue = &s.queues[queue_id];

            cudaError_t cuda_ret = cudaStreamCreate(&queue->cuda_stream);
            CUDA_CHECK_RET(cuda_ret, errh);

            if (_zc) {
                // Under Unified Virtual Addressing, host and device pointers are the same. 
                cuda_ret = cudaHostAlloc((void **) &queue->h_memory, size_per_thread, cudaHostAllocMapped);
                CUDA_CHECK_RET(cuda_ret, errh);
                queue->d_memory = queue->h_memory;
            } else {
                cuda_ret = cudaHostAlloc(&queue->h_memory, size_per_thread, cudaHostAllocDefault);
                CUDA_CHECK_RET(cuda_ret, errh);
                cuda_ret = cudaMalloc((void **) &queue->d_memory, size_per_thread);
                CUDA_CHECK_RET(cuda_ret, errh);
            }

            queue->put_index = 0;
            queue->get_index = 0;
            queue->events = new cudaEvent_t[_capacity];
            queue->batches = new PacketBatch*[_capacity];
            for (int i = 0; i < _capacity; i++) {
                cudaEventCreateWithFlags(&queue->events[i], cudaEventDisableTiming);
                queue->batches[i] = nullptr;
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

void GPUIPLookupWithCopy::push_batch(int port, PacketBatch *batch) {
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

    struct queue_state *queue = &_state->queues[_state->next_queue_put];
    uint8_t id = _state->next_queue_put;
    _state->next_queue_put = (_state->next_queue_put + 1) % _queues_per_core;

    char *h_batch_memory = queue->h_memory + ((queue->put_index * _stride) << _log_max_batch);
    char *loop_ptr = h_batch_memory;

    FOR_EACH_PACKET(batch, p) {
        const unsigned char *data = p->data();
        memcpy(loop_ptr, data + _from, _stride);
        loop_ptr += _stride;
    }

    while (unlikely(queue->batches[queue->put_index] != nullptr)) {
        if (_block) {
            // TODO fix blocking mode, it never switches to the task 
            _state->task->reschedule();
            if (unlikely(_verbose))
                click_chatter("%p{element} Congestion: Queue %u is not free!" 
                    "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this, id);
        } else {
            batch->kill();
            if (unlikely(_verbose))
                click_chatter("%p{element} Dropped %d packets: Queue %u is not free!" 
                    "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this, n, id);
            return;
        }
    }

    queue->batches[queue->put_index] = batch;

    if (_zc) {
        wrapper_iplookup(h_batch_memory, n, _cuda_blocks, _cuda_threads, queue->cuda_stream, _ip_list_gpu, _ip_list_len);
        cudaEventRecord(queue->events[queue->put_index], queue->cuda_stream);
    } else {
        char *d_batch_memory = queue->d_memory + ((queue->put_index * _stride) << _log_max_batch);
        size_t size = loop_ptr - h_batch_memory;
        cudaMemcpyAsync(d_batch_memory, h_batch_memory, size, cudaMemcpyHostToDevice, queue->cuda_stream);
        wrapper_iplookup(d_batch_memory, n, _cuda_blocks, _cuda_threads, queue->cuda_stream, _ip_list_gpu, _ip_list_len);
        if (_copyback)
            cudaMemcpyAsync(h_batch_memory, d_batch_memory, size, cudaMemcpyDeviceToHost, queue->cuda_stream);
        cudaEventRecord(queue->events[queue->put_index], queue->cuda_stream);
    }

    // if (_zc) {
    //     wrapper_iplookup(h_batch_memory, n, _cuda_blocks, _cuda_threads, _state->cuda_stream, _ip_list_gpu, _ip_list_len);
    //     cudaEventRecord(_state->events[_state->put_index], _state->cuda_stream);
    // } else {
    //     char *d_batch_memory = _state->d_memory + (_state->put_index << _log_max_batch);
    //     size_t size = loop_ptr - h_batch_memory;
    //     cudaMemcpyAsync(d_batch_memory, h_batch_memory, size, cudaMemcpyHostToDevice, _state->cuda_stream);
    //     wrapper_iplookup(d_batch_memory, n, _cuda_blocks, _cuda_threads, _state->cuda_stream, _ip_list_gpu, _ip_list_len);
    //     cudaMemcpyAsync(h_batch_memory, d_batch_memory, size, cudaMemcpyDeviceToHost, _state->cuda_stream);
    //     cudaEventRecord(_state->events[_state->put_index], _state->cuda_stream);
    // }
    
    queue->put_index = (queue->put_index + 1) % _capacity;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

/* Gets the computed batch from the GPU */
bool GPUIPLookupWithCopy::run_task(Task *task) {
    uint32_t n;
    struct rte_mbuf **pkts;
    cudaError_t status;

    struct queue_state *queue = &_state->queues[_state->next_queue_get];
    _state->next_queue_get = (_state->next_queue_get + 1) % _queues_per_core;

    if (queue->batches[queue->get_index] == nullptr) {
        task->fast_reschedule();
        return false;
    }

    status = cudaEventQuery(queue->events[queue->get_index]);
    if (status != cudaSuccess) {
        task->fast_reschedule();
        return false;
    }

    /* Copy back to the packet */
    char *h_batch_memory = queue->h_memory + ((queue->get_index * _stride) << _log_max_batch);
    char *loop_ptr = h_batch_memory;
    PacketBatch *batch = queue->batches[queue->get_index];

    uint8_t *src;
    char *data = h_batch_memory;
    src = (uint8_t *) data;
    uint32_t *tmp = (uint32_t*) h_batch_memory;

    // printf("Before Swap, Source: %02x:%02x:%02x:%02x\n",
    //         src[0], src[1], src[2], src[3]);

    // FOR_EACH_PACKET_SAFE(batch, p) {
        // WritablePacket *q = p->uniqueify();
        // unsigned char *data = q->data();
        // memcpy(data + _from, loop_ptr, _stride);

        // uint32_t temp = *tmp;
        //printf("port: %d\n", temp);
        // tmp++;
        // loop_ptr += _stride;
        // p = q;
    // }

    output_push_batch(0, batch);
    queue->batches[queue->get_index] = nullptr;
    queue->get_index = (queue->get_index + 1) % _capacity;
    return true;
}

void GPUIPLookupWithCopy::cleanup(CleanupStage cs) {
    cleanup_base(cs);
    free(_ip_list_cpu);
    cudaFree(_ip_list_gpu);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUIPLookupWithCopy)