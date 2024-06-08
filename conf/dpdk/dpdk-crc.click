/**
 * Computes the CRC and append it to the end of the packet
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/dpdk/dpdk-crc.click
 */

// Size of the mempool to allocate
define ($nb_mbuf 65535)

// Number of descriptors per ring
define ($ndesc 2048)

// number of cores receiving and processing packets
define ($maxthreads 1)

info :: DPDKInfo($nb_mbuf)

FromDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads) 
    -> SetCRC32
    
    // to check correctness, it drops packets if CRC is incorrect
    // -> CheckCRC32
    -> ToDPDKDevice(0, NDESC $ndesc, MAXTHREADS $maxthreads, BLOCKING false)

/* For debug purposes */
// Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);
