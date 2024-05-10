/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 *
 * This version uses coalescent memory areas to communicate between CPU and GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 65536)

// Number of descriptors per ring
define ($ndesc 2048)

define ($batch 1024)
define ($burst 32)
define ($minbatch 992) // has to be $batch - $burst

define ($maxthreads 2)

define ($zerocopy true)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF $(nb_mbuf))

from :: FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst) 
    -> MinBatch($minbatch, TIMER 1000) 
    -> GPUEtherMirrorCoalescent(FROM 0, TO 12, VERBOSE false, CAPACITY 4096, MAX_BATCH $batch, ZEROCOPY $zerocopy, BLOCKING false) 
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING true)

/* For debug purposes */
// Script(TYPE ACTIVE, read from.hw_dropped, wait 1s, loop);
