#ifndef CLICK_MYGPUELEMENT_HH
#define CLICK_MYGPUELEMENT_HH
#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>

CLICK_DECLS

#define MAX_BURSTS_X_QUEUE 4096


class GPUEtherMirror : public BatchElement { 
public:

    GPUEtherMirror() CLICK_COLD;

    const char *class_name() const              { return "GPUEtherMirror"; }
    const char *port_count() const              { return PORTS_1_1; }

#if HAVE_BATCH
    bool run_task(Task *);
    void push_batch(int port, PacketBatch *head);
#endif
    // int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;

protected:
    struct state {
        struct rte_gpu_comm_list *comm_list;
        uint32_t comm_list_size;
        uint32_t comm_list_put_index, comm_list_get_index;
        Task *task;
        cudaStream_t cuda_stream;
    };
    per_thread<state> _state;
};

CLICK_ENDDECLS
#endif