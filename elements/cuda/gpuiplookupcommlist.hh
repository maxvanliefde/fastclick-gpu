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
    void cleanup(CleanupStage) override CLICK_COLD;

    void push_batch(int, PacketBatch*) override CLICK_COLD;
    bool run_task(Task *) override CLICK_COLD;
    void run_timer(Timer *);

    bool cp_ip_route(String, Route *, bool, Element *);
    void print_route(Route);
    int read_from_file();

    uint32_t _ip_list_len;
    Route *_ip_list_cpu;
    RouteGPU *_ip_list_gpu;

    Vector<Route> _ip_vector_cpu;

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