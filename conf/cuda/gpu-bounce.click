/**
 * This simple configuration bounces packets from a single interface to itself
 * MAC addresses are inverted using GPU
 *
 * A minimal launch line would be:
 * sudo bin/click --dpdk -- conf/cuda/gpu-bounce.click
 */

define ($print false)
define ($batch 512)

FromDPDKDevice(0) 
    -> MinBatch($batch) 
    -> Print(INP, MAXLENGTH 12, ACTIVE $print) 
    -> GPUEtherMirror() 
    -> Print(OUT, MAXLENGTH 12, ACTIVE $print) 
    -> ToDPDKDevice(0)