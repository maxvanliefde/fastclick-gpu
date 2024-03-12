#ifndef CLICK_MYGPUELEMENT_HH
#define CLICK_MYGPUELEMENT_HH
#include <click/batchelement.hh>
#include <click/error.hh>
CLICK_DECLS

#define MAX_BURSTS_X_QUEUE 4096


class GPUEtherMirror : public SimpleBatchElement<GPUEtherMirror> { 
public:

    GPUEtherMirror() CLICK_COLD {};

    const char *class_name() const              { return "GPUEtherMirror"; }
    const char *port_count() const              { return PORTS_1_1; }



#if HAVE_BATCH
    PacketBatch *simple_action_batch(PacketBatch *);
#endif

    // int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
    // bool run_task(Task *);

    struct rte_gpu_comm_list *_comm_list;
    uint32_t _comm_list_size;
    uint32_t _comm_list_curr_index;

};

CLICK_ENDDECLS
#endif