#ifndef CLICK_MYGPUELEMENT_HH
#define CLICK_MYGPUELEMENT_HH
#include <click/batchelement.hh>
#include <click/error.hh>
CLICK_DECLS

#define MAX_BURSTS_X_QUEUE 4096


class GPUEtherMirror : public BatchElement { 
public:

    GPUEtherMirror() CLICK_COLD {};

    const char *class_name() const              { return "GPUEtherMirror"; }
    const char *port_count() const              { return PORTS_1_1; }



#if HAVE_BATCH
    // PacketBatch *simple_action_batch(PacketBatch *)
    bool run_task(Task *);
    void push_batch(int port, PacketBatch *head);
#endif

    // int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;

    struct rte_gpu_comm_list *_comm_list;
    uint32_t _comm_list_size;
    uint32_t _comm_list_put_index, _comm_list_get_index;
    Task *_task;

};

CLICK_ENDDECLS
#endif