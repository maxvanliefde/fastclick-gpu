#ifndef CLICK_GPUETHERMIRRORCOMMLIST_HH
#define CLICK_GPUETHERMIRRORCOMMLIST_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementcommlist.hh"

CLICK_DECLS

class GPUEtherMirrorCommList : public GPUElementCommList { 
public:
    GPUEtherMirrorCommList() CLICK_COLD;
    const char *class_name() const              { return "GPUEtherMirrorCommList"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
};

CLICK_ENDDECLS
#endif