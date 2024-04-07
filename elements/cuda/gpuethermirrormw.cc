#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>

#include "gpuethermirrormw.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUEtherMirrorMW::GPUEtherMirrorMW() : _state() {};

int GPUEtherMirrorMW::initialize(ErrorHandler *errh) {
    printf("Initializing with %d threads\n", master()->nthreads());
    _usable_threads.assign(master()->nthreads(), true);
    _home_thread = router()->home_thread_id(this);
    _usable_threads[_home_thread] = false;
    printf("Home thread %d\n", _home_thread);

    for (int th_id = 0; th_id < master()->nthreads(); th_id++) {
        if (!_usable_threads[th_id])
            continue;

        printf("Initializing thread %d\n", th_id);
        state &s = _state.get_value(th_id);

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);

        s.ring.initialize(1024);
    }

    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUEtherMirrorMW::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    enum rte_gpu_comm_list_status status;

    int th_id = PAINT_ANNO(batch->first());
    printf("Paint is %d\n", th_id);
    _state.get_value_for_thread(th_id).ring.insert(batch->first());
    _state.get_value_for_thread(th_id).task->reschedule();
}

bool GPUEtherMirrorMW::run_task(Task *task) {
    PacketBatch* b = reinterpret_cast<PacketBatch*>(_state->ring.extract());
    output_push_batch(0, b);
}

#endif

bool GPUEtherMirrorMW::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    unsigned int thisthread = router()->home_thread_id(this);
    bmp[thisthread] = 1;
    bmp.clear();
    for (int th_id = 0; th_id < master()->nthreads(); th_id++)
        if (th_id != _home_thread)
            bmp[th_id] = 1;
    return false;
}

void GPUEtherMirrorMW::cleanup(CleanupStage) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    auto usable_threads = get_passing_threads();

    for (int i = 0; i < _state.weight(); i++) {
        if (!usable_threads[i])
            continue;

        state &s = _state.get_value(i);

        /* Finish all the tasks polling on the GPU communication list */
        delete s.task;
        s.task = nullptr;
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirrorMW)