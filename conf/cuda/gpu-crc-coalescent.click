/**
 * Computes the CRC and append it to the end of the packet
 *
 * This version uses NVIDIA's communication lists to communicate between CPU and GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-crc-commlist.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 65535)

// Number of descriptors per ring
define ($ndesc 2048)

// A batch must contain at most 1024 packets
define ($batch 1024)
define ($burst 32)
define ($minbatch 998) // has to be $batch - $burst

// number of cores receiving and processing packets
define ($maxthreads 1)

// each thread has `queues_per_core` lists of size `capacity`
define ($capacity 128)
define ($queues_per_core 1)

// If true, GPU directly accesses CPU memory
define ($zerocopy false)

// Maximum size of a packet (defines how many bytes are copied per packet); better if cache aligned
define ($max_pkt_size 64)



info :: DPDKInfo($nb_mbuf)

FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BURST $burst, VERBOSE 10, NUMA 0) 
    -> MinBatch($minbatch, TIMER 1000) 
    -> GPUSetCRC32Coalescent(MAX_PKT_SIZE $max_pkt_size, VERBOSE false, CAPACITY $capacity, MAX_BATCH $batch, QUEUES_PER_CORE $queues_per_core, ZEROCOPY $zerocopy) 

    // to check correctness, it drops packets if CRC is incorrect
    -> CheckCRC32
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING false)

/* For debug purposes */
// Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);
