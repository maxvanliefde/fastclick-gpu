#ifndef CLICK_GPUETHERMIRRORCOALESCENTGRAPH_HH
#define CLICK_GPUETHERMIRRORCOALESCENTGRAPH_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementcoalescentgraph.hh"

CLICK_DECLS

class GPUEtherMirrorCoalescentGraph : public GPUElementCoalescentGraph { 
public:
    GPUEtherMirrorCoalescentGraph() CLICK_COLD;
    const char *class_name() const              { return "GPUEtherMirrorCoalescentGraph"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
};

CLICK_ENDDECLS
#endif