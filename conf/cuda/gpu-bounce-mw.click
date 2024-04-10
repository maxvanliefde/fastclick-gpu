define ($batch 512)
define ($maxthread 8)

info :: DPDKInfo(NB_SOCKET_MBUF 0, NB_SOCKET_MBUF 262143)

Script(TYPE ACTIVE, read load, read info.pool_count, wait 1s, loop);

StaticThreadSched(p 15)
StaticThreadSched(gpu 15)

FromDPDKDevice(0, NDESC 1024, BURST 32, MAXTHREADS $maxthread, VERBOSE 10) 
    -> CPUPaint()
    // -> Print(INP, MAXLENGTH 12, ACTIVE true) 
    -> p :: Pipeliner(BURST 32, BLOCKING true)
    -> gpu :: GPUEtherMirrorMW(p, VERBOSE true) 
    // -> Print(OUT, MAXLENGTH 12, ACTIVE true) 
    -> ToDPDKDevice(0, NDESC 1024, MAXTHREADS $maxthread, VERBOSE 10)