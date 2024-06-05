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

define ($nb_mbuf 65535)
define ($batch 1024)
define ($burst 32)

define ($maxthread 2)       // up to 6

info :: DPDKInfo($nb_mbuf)

define ($masterthread 6)
StaticThreadSched(p $masterthread)

fromdpdk :: FromDPDKDevice(0, NDESC 2048, BURST $burst, MAXTHREADS $maxthread, VERBOSE 10) 
todpdk :: ToDPDKDevice(0, NDESC 2048, MAXTHREADS $maxthread, VERBOSE 10) 

fromdpdk -> CPUPaint()
    -> p :: Pipeliner(BURST $burst, BLOCKING false, ALWAYS_UP false, PREFETCH true)
    -> ConstantBatch($batch, TIMER 1000)
    -> gpu :: GPUEtherMirrorCoalescent(MAX_BATCH $batch, VERBOSE true, CAPACITY 1024, QUEUES_PER_CORE 2)
    -> switch :: PaintSwitch()

switch[0] -> p0 :: Pipeliner(THREAD 0) -> [0]todpdk
switch[1] -> p1 :: Pipeliner(THREAD 1) -> [0]todpdk
switch[2] -> p2 :: Pipeliner(THREAD 2) -> [0]todpdk
switch[3] -> p3 :: Pipeliner(THREAD 3) -> [0]todpdk
switch[4] -> p4 :: Pipeliner(THREAD 4) -> [0]todpdk
switch[5] -> p5 :: Pipeliner(THREAD 5) -> [0]todpdk

/* For debug purposes */
// Script(TYPE ACTIVE, read info.pool_count, read p.count, read p0.count, read p1.count, read p2.count, read p3.count, wait 1s, loop);
