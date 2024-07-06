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

    bool get_spawning_threads(Bitvector& b, bool isoutput, int port) override;
    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;

    void push_batch(int, PacketBatch*) override CLICK_COLD;
    bool run_task(Task*) override CLICK_COLD;
    void run_timer(Timer *);

    bool cp_ip_route(String, Route *, bool, Element *);
    void print_route(Route);
    int read_from_file(uint8_t);

    uint32_t _ip_list_len;
    Route *_ip_list_cpu;
    RouteGPU *_ip_list_gpu;

    Vector<Route> _ip_vector_cpu;

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
    uint8_t _lookup_table;
    int _cuda_threads, _cuda_blocks;
};

CLICK_ENDDECLS
#endif