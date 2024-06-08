#ifndef CLICK_GPUELEMENTCOMMLIST_HH
#define CLICK_GPUELEMENTCOMMLIST_HH
#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>

typedef void (base_wrapper_persistent_kernel)(struct rte_gpu_comm_list *comm_list, uint32_t comm_list_size, int cuda_blocks, int cuda_threads, cudaStream_t stream);

CLICK_DECLS

class GPUElementCommList : public BatchElement { 
public:

    GPUElementCommList() CLICK_COLD;

    const char *class_name() const              { return "GPUElementCommList"; }
    const char *port_count() const              { return PORTS_1_1; }
    const char *flow_code()  const override     { return COMPLETE_FLOW;} 
    const char *processing() const              { return PUSH;}

    bool get_spawning_threads(Bitvector& b, bool isoutput, int port) override;
    int configure_base(Vector<String> &, ErrorHandler *) CLICK_COLD;
    int initialize_base(ErrorHandler *, base_wrapper_persistent_kernel) CLICK_COLD;
    void cleanup_base(CleanupStage) CLICK_COLD;

    void push_batch(int port, PacketBatch *head);
    bool run_task(Task *);
    void run_timer(Timer *);

protected:
    struct comm_list_state {
        struct rte_gpu_comm_list *comm_list;
        uint32_t comm_list_size;
        uint32_t comm_list_put_index, comm_list_get_index;
        cudaStream_t cuda_stream;
    };

    struct state {
        struct comm_list_state *comm_lists;
        uint8_t next_list_get, next_list_put;   // round robin
        Task *task;
        Timer *timer;
    };
    per_thread<state> _state;

    Bitvector _usable_threads;

    /* Parameters */
    uint32_t _capacity;
    uint16_t _blocks_per_q;
    uint16_t _max_batch;
    uint8_t _lists_per_core;
    bool _persistent_kernel;
    bool _block;
    bool _verbose;
};

CLICK_ENDDECLS
#endif