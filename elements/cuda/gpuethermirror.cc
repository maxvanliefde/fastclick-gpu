#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuethermirror.hh"

CLICK_DECLS

GPUEtherMirror::GPUEtherMirror() {};

int GPUEtherMirror::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .consume() < 0)
        return -1;

    return 0;
}

int GPUEtherMirror::initialize(ErrorHandler *errh) {
    return initialize_base(errh, wrapper_ether_mirror_persistent);
}

void GPUEtherMirror::cleanup(CleanupStage cs) {
    cleanup_base(cs);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirror)