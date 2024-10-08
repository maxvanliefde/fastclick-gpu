#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuethermirrorcommlist.hh"

CLICK_DECLS

GPUEtherMirrorCommList::GPUEtherMirrorCommList() {};

int GPUEtherMirrorCommList::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .complete() < 0)
        return -1;

    return 0;
}

int GPUEtherMirrorCommList::initialize(ErrorHandler *errh) {
    return initialize_base(errh, wrapper_ether_mirror_persistent);
}

void GPUEtherMirrorCommList::cleanup(CleanupStage cs) {
    cleanup_base(cs);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirrorCommList)