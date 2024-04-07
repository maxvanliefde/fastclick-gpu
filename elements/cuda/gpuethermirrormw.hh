#ifndef CLICK_MYGPUELEMENTMW_HH
#define CLICK_MYGPUELEMENTMW_HH
#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <click/ring.hh>
#include <cuda_runtime.h>

CLICK_DECLS

class GPUEtherMirrorMW : public BatchElement { 
public:

    GPUEtherMirrorMW() CLICK_COLD;

    const char *class_name() const              { return "GPUEtherMirrorMW"; }
    const char *port_count() const              { return PORTS_1_1; }
    const char *flow_code()  const override     { return COMPLETE_FLOW;} 
    const char *processing() const              { return PUSH;}

#if HAVE_BATCH
    bool run_task(Task *);
    void push_batch(int port, PacketBatch *head);
#endif
    // int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
    bool get_spawning_threads(Bitvector&, bool, int);

protected:
    struct state {
        Task *task;
        DynamicRing<Packet*> ring;
    };
    per_thread<state> _state;
    Bitvector _usable_threads;
    int _home_thread;
    int next = 0;
};

CLICK_ENDDECLS
#endif