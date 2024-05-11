#ifndef CLICK_GPUIPLOOKUP_HH
#define CLICK_GPUIPLOOKUP_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>
#include <click/cuda/kerneliplookup.hh>

#include "gpuelementcommlist.hh"
#include "route.hh"

CLICK_DECLS

class GPUIPLookup : public GPUElementCommList { 
public:
    GPUIPLookup() CLICK_COLD;
    const char *class_name() const              { return "GPUIPLookup"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    bool run_task(Task *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;

    bool cp_ip_route(String, Route *, bool, Element *);
    void print_route(Route);

    uint32_t _ip_list_len;
    Route *_ip_list_cpu;
    RouteGPU *_ip_list_gpu;
};

CLICK_ENDDECLS
#endif