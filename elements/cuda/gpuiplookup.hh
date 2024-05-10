#ifndef CLICK_GPUIPLOOKUP_HH
#define CLICK_GPUIPLOOKUP_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>
#include <click/cuda/kerneliplookup.hh>

#include "gpuelementcommlist.hh"

CLICK_DECLS

struct Route {
    IPAddress addr;
    IPAddress mask;
    IPAddress gw;
    int32_t port;
    int32_t extra;
};

class GPUIPLookup : public GPUElementCommList { 
public:
    GPUIPLookup() CLICK_COLD;
    const char *class_name() const              { return "GPUIPLookup"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    bool run_task(Task *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;

    uint32_t _ip_list_len;
    Route *_ip_list_cpu;
    RouteGPU *_ip_list_gpu;
};

CLICK_ENDDECLS
#endif