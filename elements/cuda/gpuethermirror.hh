#ifndef CLICK_GPUETHERMIRROR_HH
#define CLICK_GPUETHERMIRROR_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelement.hh"

CLICK_DECLS

class GPUEtherMirror : public GPUElement { 
public:
    GPUEtherMirror() CLICK_COLD;
    const char *class_name() const              { return "GPUEtherMirror"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
};

CLICK_ENDDECLS
#endif