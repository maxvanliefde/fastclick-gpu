#ifndef CLICK_GPUETHERMIRRORCOALESCENT_HH
#define CLICK_GPUETHERMIRRORCOALESCENT_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementcoalescent.hh"

CLICK_DECLS

class GPUEtherMirrorCoalescent : public GPUElementCoalescent { 
public:
    GPUEtherMirrorCoalescent() CLICK_COLD;
    const char *class_name() const              { return "GPUEtherMirrorCoalescent"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
};

CLICK_ENDDECLS
#endif