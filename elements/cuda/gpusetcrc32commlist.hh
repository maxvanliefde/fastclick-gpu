#ifndef CLICK_GPUSETCRC32_HH
#define CLICK_GPUSETCRC32_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementcommlist.hh"

CLICK_DECLS

class GPUSetCRC32CommList : public GPUElementCommList { 
public:
    GPUSetCRC32CommList() CLICK_COLD;
    const char *class_name() const              { return "GPUSetCRC32CommList"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
    void push_batch(int port, PacketBatch *head);


    uint32_t _crc_table[256];
    uint32_t *_gpu_table;
};

CLICK_ENDDECLS
#endif