/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 8195)

// Number of descriptors per ring
define ($ndesc 2048)

// A batch must contain at most 1024 packets
define ($batch 1024)
define ($burst 32)
define ($minbatch 992) // has to be $batch - $burst

define ($maxthreads 1)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF $(nb_mbuf))

from :: FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst) 
    -> MinBatch($minbatch, TIMER 10) 
    // -> Print(INP)
    -> GPUEtherMirrorWithCopy(FROM 0, TO 12, VERBOSE true, CAPACITY 4096, MAX_BATCH $batch) 
    // -> Print(OUT)
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING true)

/* For debug purposes */
Script(TYPE ACTIVE, read from.hw_dropped, wait 1s, loop);
