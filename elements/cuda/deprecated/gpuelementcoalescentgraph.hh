#ifndef CLICK_GPUELEMENTCOALESCENTGRAPH_HH
#define CLICK_GPUELEMENTCOALESCENTGRAPH_HH
#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <cuda.h>

CLICK_DECLS

class GPUElementCoalescentGraph : public BatchElement { 
public:

    GPUElementCoalescentGraph() CLICK_COLD;

    const char *class_name() const              { return "GPUElementCoalescentGraph"; }
    const char *port_count() const              { return PORTS_1_1; }
    const char *flow_code()  const override     { return COMPLETE_FLOW;} 
    const char *processing() const              { return PUSH;}

    bool get_spawning_threads(Bitvector& b, bool isoutput, int port) override;
    int configure_base(Vector<String> &, ErrorHandler *) CLICK_COLD;
    int initialize_base(ErrorHandler *) CLICK_COLD;
    void cleanup_base(CleanupStage) CLICK_COLD;

    void push_batch(int port, PacketBatch *head);
    bool run_task(Task *);
    void run_timer(Timer *);

protected:
    struct queue_state {
        char *h_memory, *d_memory;
        uint32_t put_index, get_index;
        uint32_t *d_wait_signals, *d_post_signals;
        PacketBatch **batches;
        uint32_t *n_packets;
        cudaStream_t cuda_stream, cuda_signal_stream;
        cudaGraph_t graph;
        cudaGraphExec_t instance;
    };
    struct state {
        struct queue_state *queues;
        uint8_t next_queue_put, next_queue_get;
        Task *task;
        Timer *timer;
    };
    per_thread<state> _state;

    Bitvector _usable_threads;

    /* Parameters */
    uint16_t _from, _to, _stride;
    uint32_t _capacity;
    uint16_t _max_batch;
    uint8_t _log_max_batch;
    bool _block;
    bool _verbose;
    bool _zc;
    bool _copyback;
    uint8_t _queues_per_core;
    int _cuda_threads, _cuda_blocks;
};

CLICK_ENDDECLS
#endif