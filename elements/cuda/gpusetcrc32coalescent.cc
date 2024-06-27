#include <click/config.h>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <click/cuda/cuda_crc32.hh>
#include <click/crc32.h>
#include <rte_ether.h>


#include "gpusetcrc32coalescent.hh"

CLICK_DECLS

GPUSetCRC32Coalescent::GPUSetCRC32Coalescent() {};

int GPUSetCRC32Coalescent::configure(Vector<String> &conf, ErrorHandler *errh) {
    _to = -1;
    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .read_p("MAX_PKT_SIZE", _max_pkt_size)
        .complete() < 0)
        return -1;

    // allocate memory for size of packet, content and result
    _stride = _max_pkt_size + sizeof(uint32_t) + RTE_ETHER_CRC_LEN;

    // not useful for this particular element
    _from = 0;
    _to = 0;

    return 0;
}

int GPUSetCRC32Coalescent::initialize(ErrorHandler *errh) {
    /* Generate table and copy it on GPU */
    gen_crc_table(_crc_table);
    CUDA_CHECK(cudaMalloc(&_gpu_table, CRC_TABLE_LENGTH * sizeof(uint32_t)), errh);
    CUDA_CHECK(cudaMemcpy(_gpu_table, _crc_table, CRC_TABLE_LENGTH * sizeof(uint32_t), cudaMemcpyHostToDevice), errh);

    return initialize_base(errh);
}


/* Sends the batch to the GPU */
void GPUSetCRC32Coalescent::push_batch(int port, PacketBatch *batch) {
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

    // drop too large packets
    EXECUTE_FOR_EACH_PACKET_DROPPABLE([this] (Packet* p) -> Packet* {
            if (p->length() > _max_pkt_size) {
                return nullptr;
            }
            return p->put(4);
        }, batch, [this] (Packet* p) {
            click_chatter("%p{element} Warning: received a packet of size %d, but the max packet size is %d. "
                "Dropped it", this, p->length(), _max_pkt_size);
        });

    // check if packets remain
    if (batch == 0) {
        return;
    }
    n = batch->count();

    // copy sizes
    uint32_t size; 
    FOR_EACH_PACKET(batch, p) {
        size = p->length();
        memcpy(loop_ptr, &size, sizeof(uint32_t));
        loop_ptr += sizeof(uint32_t);
    }

    // copy data
    FOR_EACH_PACKET(batch, p) {
        memcpy(loop_ptr, p->data(), p->length());
        loop_ptr += _max_pkt_size;
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
        wrapper_crc32_coalescent(h_batch_memory, n, _max_pkt_size, _gpu_table, _cuda_blocks, _cuda_threads, queue->cuda_stream);
        cudaEventRecord(queue->events[queue->put_index], queue->cuda_stream);
    } else {
        char *d_batch_memory = queue->d_memory + ((queue->put_index * _stride) << _log_max_batch);
        size_t size = loop_ptr - h_batch_memory;
        cudaMemcpyAsync(d_batch_memory, h_batch_memory, size, cudaMemcpyHostToDevice, queue->cuda_stream);
        wrapper_crc32_coalescent(d_batch_memory, n, _max_pkt_size, _gpu_table, _cuda_blocks, _cuda_threads, queue->cuda_stream);
        cudaMemcpyAsync(loop_ptr, d_batch_memory + size, RTE_ETHER_CRC_LEN * n, cudaMemcpyDeviceToHost, queue->cuda_stream);
        cudaEventRecord(queue->events[queue->put_index], queue->cuda_stream);
    }
    
    queue->put_index = (queue->put_index + 1) % _capacity;
    if (!_state->task->scheduled())
        _state->task->reschedule();
}

/* Gets the computed batch from the GPU */
bool GPUSetCRC32Coalescent::run_task(Task *task) {
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

    PacketBatch *batch = queue->batches[queue->get_index];

    /* Copy back CRC to the packet */
    char *h_batch_memory = queue->h_memory + ((queue->get_index * _stride) << _log_max_batch);
    char *loop_ptr = h_batch_memory + batch->count() * (_max_pkt_size + sizeof(uint32_t));
    FOR_EACH_PACKET_SAFE(batch, p) {
        WritablePacket *q = p->uniqueify();
        uint32_t size = p->length() - RTE_ETHER_CRC_LEN;
        memcpy(q->data() + size, loop_ptr, RTE_ETHER_CRC_LEN);
        loop_ptr += RTE_ETHER_CRC_LEN;
        p = q;
    }

    output_push_batch(0, batch);
    queue->batches[queue->get_index] = nullptr;
    queue->get_index = (queue->get_index + 1) % _capacity;
    return true;
}



void GPUSetCRC32Coalescent::cleanup(CleanupStage cs) {
    cleanup_base(cs);
    cudaFree(_crc_table);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUSetCRC32Coalescent)