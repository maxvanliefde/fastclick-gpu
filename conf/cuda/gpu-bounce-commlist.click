/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 * 
 * This version uses NVIDIA's communication lists to communicate between CPU and GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 65535)

// Number of descriptors per ring
define ($ndesc 2048)

// A batch must contain at most 1024 packets
define ($batch 128)
define ($burst 32)
define ($minbatch 96) // has to be $batch - $burst

define ($maxthreads 3)

define ($capacity 128)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF $(nb_mbuf))

FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst) 
    -> MinBatch($minbatch, TIMER 1000) 
    -> GPUEtherMirrorCommList(VERBOSE false, CAPACITY $capacity, MAX_BATCH $batch, LISTS_PER_CORE 2) 
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING false)

/* For debug purposes */
// Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);
