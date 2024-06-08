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
define ($batch 256)
define ($burst 32)
define ($minbatch 224) // has to be $batch - $burst

// number of cores receiving and processing packets
define ($maxthreads 1)

// each thread has `lists_per_core` lists of size `capacity`
define ($capacity 256)
define ($lists_per_core 1)

// If true, NIC stores packets directly in GPU memory
define ($gpudirect_rdma false)

// Control Flow
define ($persistent_kernel false)


info :: DPDKInfo(MEMPOOL_GPU $gpudirect_rdma)

FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst) 
    -> MinBatch($minbatch, TIMER 1000) 
    -> GPUEtherMirrorCommList(VERBOSE false, PERSISTENT_KERNEL $persistent_kernel, CAPACITY $capacity, MAX_BATCH $batch, LISTS_PER_CORE $lists_per_core) 
    // -> Print(OUT, MAXLENGTH 12)
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING false)

/* For debug purposes */
// Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);
