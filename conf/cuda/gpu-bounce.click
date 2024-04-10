/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 131071)

// Number of descriptors per ring
define ($ndesc 2048)

// A batch must contain at most 1024 packets
define ($batch 128)
define ($burst 32)
define ($minbatch 96) // has to be $batch - $burst

define ($maxthreads 8)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF $(nb_mbuf))

FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst) 
    -> MinBatch($minbatch, TIMER 10) 
    -> GPUEtherMirror(VERBOSE true, CAPACITY 4096, MAX_BATCH $batch, THREAD_BLOCKS_PER_QUEUE 1) 
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING true)

/* For debug purposes */
Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);
