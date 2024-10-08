#include <click/config.h>
#include <click/cuda/cuda_ethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <click/algorithm.hh>

#include "gpuelementcoalescentgraph.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUElementCoalescentGraph::GPUElementCoalescentGraph() : _state(), _capacity(4096), _max_batch(1024), _block(false), _verbose(false), _zc(true), _copyback(false) {};

int GPUElementCoalescentGraph::configure_base(Vector<String> &conf, ErrorHandler *errh) {
    if (Args(conf, this, errh)
        .bind(conf)
        .read_p("FROM", _from)
        .read_p("TO", _to)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .read_p("ZEROCOPY", _zc)
        .read_p("COPYBACK", _copyback)
        .read_p("QUEUES_PER_CORE", _queues_per_core)
        .consume() < 0) 
    {
        return -1;
    }

    if (_zc && _copyback) {
        errh->warning("Warning: COPYBACK has no impact when ZEROCOPY is enabled");
    }

    if (_queues_per_core == 0) {
        errh->error("Given QUEUES_PER_CORE = 0 but it has to be at least 1");
        return -1;
    }


    /* Compute stride */
    if (_to <= _from) {
        return errh->error("TO should be greather than FROM, but from = %d and to = %d", _from, _to);
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

int GPUElementCoalescentGraph::initialize_base(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();

    size_t size_per_thread = ((size_t) _stride *  _capacity) << _log_max_batch;

    if (_verbose) {
        const char *str = "%p{element} Initialization:\n"
        "* from = %d, to = %d => %d bytes will be read per packet\n"
        "* a batch contains maximum %d packets\n"
        "* capacity of each queue is %d\n"
        "* %ld B will be allocated per queue\n";
        click_chatter(str, this, _from, _to, _stride, _max_batch, _capacity, size_per_thread);
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
            cuda_ret = cudaStreamCreate(&queue->cuda_signal_stream);
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
            queue->batches = new PacketBatch*[_capacity];

            size_t signals_size = sizeof(uint32_t) * _capacity;
            CUDA_CHECK_RET(cudaMalloc(&queue->d_wait_signals, signals_size), errh);
            CUDA_CHECK_RET(cudaMemset(queue->d_wait_signals, 0, signals_size), errh);
            CUDA_CHECK_RET(cudaMalloc(&queue->d_post_signals, signals_size), errh);
            CUDA_CHECK_RET(cudaMemset(queue->d_post_signals, 0, signals_size), errh);
            CUDA_CHECK_RET(cudaMalloc(&queue->n_packets, signals_size), errh);
            for (int i = 0; i < _capacity; i++) {
                queue->batches[i] = nullptr;
            }

            // define the graph
            CUDA_CHECK_RET(cudaGraphCreate(&queue->graph, 0), errh);
            cudaGraphNode_t node;
            cudaGraphConditionalHandle handle;
            CUDA_CHECK_RET(cudaGraphConditionalHandleCreate(&handle, queue->graph, 1U, 0), errh);
            cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
            cParams.conditional.handle = handle;
            cParams.conditional.type = cudaGraphCondTypeWhile;
            cParams.conditional.size = 1;
            CUDA_CHECK_RET(cudaGraphAddNode(&node, queue->graph, NULL, 0, &cParams), errh);

            cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
            cudaStream_t captureStream;
            cudaStreamCreate(&captureStream);
            cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed);

            CUdeviceptr wait_signal, post_signal;
            uint32_t *n_packets;
            char *batch_memory;
            for (uint32_t i_item = 0; i_item < _capacity; i_item++) {
                wait_signal = reinterpret_cast<CUdeviceptr>(&queue->d_wait_signals[i_item]);
                post_signal = reinterpret_cast<CUdeviceptr>(&queue->d_post_signals[i_item]);
                n_packets = &queue->n_packets[i_item];
                batch_memory = queue->h_memory + (queue->get_index << _log_max_batch);

                cuStreamWaitValue32(queue->cuda_stream, wait_signal, 1, CU_STREAM_WAIT_VALUE_EQ);
                wrapper_ether_mirror_graph(batch_memory, n_packets, _cuda_blocks, _cuda_threads, queue->cuda_stream);
                cuStreamWriteValue32(queue->cuda_stream, wait_signal, 2, CU_STREAM_WRITE_VALUE_DEFAULT);
            }
            cudaStreamEndCapture(queue->cuda_stream, nullptr);
            cudaStreamDestroy(captureStream);

            cudaGraphNode_t *nodes = NULL;
            size_t numNodes = 0;
            (cudaGraphGetNodes(queue->graph, nodes, &numNodes));
            printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
                    numNodes);

            cudaGraphInstantiate(&queue->instance, queue->graph);

            // first launch is the longest
            // cudaGraphLaunch(queue->instance, queue->cuda_stream);
        }
    }

    if (_verbose) {
        click_chatter("%p{element}: %d usable threads, %d queue per thread, %d batch per queue.",
            this,
            _usable_threads.weight(), _queues_per_core, _capacity
        );
        for (int th_id = 0; th_id < master()->nthreads(); th_id++)
            if (_usable_threads[th_id])
                click_chatter("%p{element}: Pipeline in thread %d", this, th_id);
    }

    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUElementCoalescentGraph::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    cudaError_t ret;

    uint32_t n = batch->count();

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

    if (queue->put_index == 0)
        cudaGraphLaunch(queue->instance, queue->cuda_stream);

    char *h_batch_memory = queue->h_memory + (queue->put_index << _log_max_batch);
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
    cudaMemcpyAsync(&queue->n_packets[queue->put_index], &n, sizeof(uint32_t), cudaMemcpyHostToDevice, queue->cuda_signal_stream);
    CUdeviceptr wait_signal = reinterpret_cast<CUdeviceptr>(&queue->d_wait_signals[queue->put_index]);
    uint32_t flag = 1;
    // cudaMemcpyAsync(&queue->d_wait_signals[queue->put_index], &flag, sizeof(uint32_t), cudaMemcpyHostToDevice, queue->cuda_signal_stream);

    cuStreamWriteValue32(queue->cuda_signal_stream, wait_signal, 1, CU_STREAM_WRITE_VALUE_DEFAULT); // mem barrier

    queue->put_index = (queue->put_index + 1) % _capacity;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

/* Gets the computed batch from the GPU */
bool GPUElementCoalescentGraph::run_task(Task *task) {
    uint32_t n;
    struct rte_mbuf **pkts;
    cudaError_t status;

    struct queue_state *queue = &_state->queues[_state->next_queue_get];
    _state->next_queue_get = (_state->next_queue_get + 1) % _queues_per_core;

    if (queue->batches[queue->get_index] == nullptr) {
        task->fast_reschedule();
        return false;
    }

    uint32_t signal_value;
    uint32_t *signal = &queue->d_wait_signals[queue->get_index];
    cudaMemcpyAsync(&signal_value, signal, sizeof(uint32_t), cudaMemcpyDeviceToHost, queue->cuda_signal_stream);
    cudaStreamSynchronize(queue->cuda_signal_stream);
    if (signal_value != 2) {
        task->fast_reschedule();
        return false;
    }

    /* Copy back to the packet */
    char *h_batch_memory = queue->h_memory + (queue->get_index << _log_max_batch);
    char *loop_ptr = h_batch_memory;
    PacketBatch *batch = queue->batches[queue->get_index];
    FOR_EACH_PACKET_SAFE(batch, p) {
        WritablePacket *q = p->uniqueify();
        unsigned char *data = q->data();
        memcpy(data + _from, loop_ptr, _stride);
        loop_ptr += _stride;
        p = q;
    }

    output_push_batch(0, batch);
    cuStreamWriteValue32(queue->cuda_signal_stream, reinterpret_cast<CUdeviceptr>(signal), 0, CU_STREAM_WRITE_VALUE_DEFAULT);
    queue->batches[queue->get_index] = nullptr;
    queue->get_index = (queue->get_index + 1) % _capacity;
    return true;
}

void GPUElementCoalescentGraph::run_timer(Timer *timer) {
    _state->task->reschedule();
    timer->schedule_now();
}

#endif

bool GPUElementCoalescentGraph::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    return true;
}


/* Cleans all structures */
void GPUElementCoalescentGraph::cleanup_base(CleanupStage) {
    ErrorHandler *errh = ErrorHandler::default_handler();

    for (int i = 0; i < _state.weight(); i++) {
        if (!_usable_threads[i])
            continue;

        state &s = _state.get_value(i);

        /* Finish all the tasks polling on the GPU communication list */
        delete s.task;
        s.task = nullptr;

        /* Destroy CUDA resources */
        for (uint8_t queue_id = 0; queue_id < _queues_per_core; queue_id++) {
            struct queue_state *queue = &s.queues[queue_id];

            cudaStreamDestroy(queue->cuda_stream);
            cudaFreeHost(&queue->h_memory);
            if (!_zc) cudaFree(&queue->d_memory);
            // for (int i = 0; i < _capacity; i++) {
            //     cudaEventDestroy(queue->events[i]);
            // }
            // delete[] queue->events;
            delete[] queue->batches;
        }
        delete[] s.queues;
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUElementCoalescentGraph)