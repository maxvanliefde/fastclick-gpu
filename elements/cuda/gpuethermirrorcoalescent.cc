#include <click/config.h>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuethermirrorcoalescent.hh"

CLICK_DECLS

GPUEtherMirrorCoalescent::GPUEtherMirrorCoalescent() {};

int GPUEtherMirrorCoalescent::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .complete() < 0)
        return -1;

    return 0;
}

int GPUEtherMirrorCoalescent::initialize(ErrorHandler *errh) {
    return initialize_base(errh);
}

void GPUEtherMirrorCoalescent::cleanup(CleanupStage cs) {
    cleanup_base(cs);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirrorCoalescent)