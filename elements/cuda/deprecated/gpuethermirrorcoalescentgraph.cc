#include <click/config.h>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>
#include <rte_ether.h>

#include "gpuethermirrorcoalescentgraph.hh"

CLICK_DECLS

GPUEtherMirrorCoalescentGraph::GPUEtherMirrorCoalescentGraph() {};

int GPUEtherMirrorCoalescentGraph::configure(Vector<String> &conf, ErrorHandler *errh) {
    _from = 0;
    _to = RTE_ETHER_ADDR_LEN * 2;

    if (configure_base(conf, errh) != 0)
        return -1;

    if (Args(conf, this, errh)
        .complete() < 0)
        return -1;

    return 0;
}

int GPUEtherMirrorCoalescentGraph::initialize(ErrorHandler *errh) {
    return initialize_base(errh);
}

void GPUEtherMirrorCoalescentGraph::cleanup(CleanupStage cs) {
    cleanup_base(cs);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirrorCoalescentGraph)