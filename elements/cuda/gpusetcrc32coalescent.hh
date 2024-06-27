#ifndef CLICK_GPUSETCRC32COALSCENT_HH
#define CLICK_GPUSETCRC32COALSCENT_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementcoalescent.hh"

CLICK_DECLS

class GPUSetCRC32Coalescent : public GPUElementCoalescent { 
public:
    GPUSetCRC32Coalescent() CLICK_COLD;
    const char *class_name() const              { return "GPUSetCRC32Coalescent"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void push_batch(int port, PacketBatch *batch);
    bool run_task(Task *);
    void cleanup(CleanupStage) override CLICK_COLD;

private:
    uint32_t _crc_table[256];
    uint32_t *_gpu_table;
    
    uint32_t _max_pkt_size; 

};

CLICK_ENDDECLS
#endif