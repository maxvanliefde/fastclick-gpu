#ifndef CLICK_GPUETHERMIRRORWITHCOPY_HH
#define CLICK_GPUETHERMIRRORWITHCOPY_HH

#include <click/batchelement.hh>
#include <click/error.hh>
#include <click/sync.hh>
#include <cuda_runtime.h>
#include <click/glue.hh>

#include "gpuelementwithcopy.hh"

CLICK_DECLS

class GPUEtherMirrorWithCopy : public GPUElementWithCopy { 
public:
    GPUEtherMirrorWithCopy() CLICK_COLD;
    const char *class_name() const              { return "GPUEtherMirrorWithCopy"; }

    int configure(Vector<String> &, ErrorHandler *) override CLICK_COLD;
    int initialize(ErrorHandler *) override CLICK_COLD;
    void cleanup(CleanupStage) override CLICK_COLD;
};

CLICK_ENDDECLS
#endif