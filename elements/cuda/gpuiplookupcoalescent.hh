#ifndef CLICK_GPUIPLOOKUPWITHCOPY_HH
#define CLICK_GPUIPLOOKUPWITHCOPY_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>
#include <click/cuda/cuda_iplookup.hh>

#include "gpuelementcoalescent.hh"
#include "route.hh"

CLICK_DECLS

class GPUIPLookupWithCopy : public GPUElementCoalescent { 
public:
    GPUIPLookupWithCopy() CLICK_COLD;
    const char *class_name() const              { return "GPUIPLookupWithCopy"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
    void push_batch(int, PacketBatch*) override CLICK_COLD;
    bool run_task(Task*) override CLICK_COLD;

    bool cp_ip_route(String, Route *, bool, Element *);
    void print_route(Route);

    uint32_t _ip_list_len;
    Route *_ip_list_cpu;
    RouteGPU *_ip_list_gpu;
};

CLICK_ENDDECLS
#endif