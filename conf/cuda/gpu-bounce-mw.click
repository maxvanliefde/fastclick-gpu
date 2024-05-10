/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 *
 * A single CPU cores handles the communication with the GPU,
 * While others are worker threads
 * This version uses coalescent memory areas to communicate between CPU and GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

define ($nb_mbuf 65536)
define ($maxthread 4)       // up to 4
define ($batch 1024)
define ($burst 32)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF $nb_mbuf)

StaticThreadSched(p 9)
StaticThreadSched(gpu 9)

fromdpdk :: FromDPDKDevice(0, NDESC 2048, BURST $burst, MAXTHREADS $maxthread, VERBOSE 10) 
todpdk :: ToDPDKDevice(0, NDESC 2048, MAXTHREADS $maxthread, VERBOSE 10) 

fromdpdk -> CPUPaint()
    -> p :: Pipeliner(BURST $burst, BLOCKING false, ALWAYS_UP false, PREFETCH true)
    -> ConstantBatch($batch, TIMER 1000)
    -> gpu :: GPUEtherMirrorCoalescent(FROM 0, TO 12, MAX_BATCH $batch, VERBOSE true, CAPACITY 1024, ZEROCOPY true)
    -> switch :: PaintSwitch()

switch[0] -> Discard()
switch[1] -> p0 :: Pipeliner(THREAD 1) -> [0]todpdk
switch[2] -> Discard()
switch[3] -> p1 :: Pipeliner(THREAD 3) -> [0]todpdk
switch[4] -> Discard()
switch[5] -> p2 :: Pipeliner(THREAD 5) -> [0]todpdk
switch[6] -> Discard()
switch[7] -> p3 :: Pipeliner(THREAD 7) -> [0]todpdk

/* For debug purposes */
Script(TYPE ACTIVE, read p.count, read p0.count, read p1.count, read p2.count, read p3.count, wait 1s, loop);
