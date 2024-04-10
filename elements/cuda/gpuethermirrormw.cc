#include <click/config.h>
#include <click/cuda/kernelethermirror.hh>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuethermirrormw.hh"

CLICK_DECLS
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

GPUEtherMirrorMW::GPUEtherMirrorMW() : _state(), _capacity(4096), _block(false), _verbose(false), _max_batch(32) {};

int GPUEtherMirrorMW::configure(Vector<String> &conf, ErrorHandler *errh) {
    if (Args(conf, this, errh)
        .read_p("PIPELINER", ElementCastArg("Pipeliner"), _pipeliner)
        .read_p("CAPACITY", _capacity)
        .read_p("MAX_BATCH", _max_batch)
        .read_p("BLOCKING", _block)
        .read("VERBOSE", _verbose)
        .complete() < 0) 
    {
        return -1;
    }

    if (_max_batch > RTE_GPU_COMM_LIST_PKTS_MAX) {
        errh->error("Given MAX_BATCH = %d but the maximum is %d", _max_batch, RTE_GPU_COMM_LIST_PKTS_MAX);
        return -1;
    }

    return 0;
}


int GPUEtherMirrorMW::initialize(ErrorHandler *errh) {
    _workers = _pipeliner->get_pushing_threads();
    _master = router()->home_thread_id(this);
    if (_verbose) {
        click_chatter("%p{element} " 
            "Initializing with %d workers threads", this, _workers.weight());
    }

    _comm_list_size = _capacity;
    _comm_list = rte_gpu_comm_create_list(0, _comm_list_size);   // todo generify GPU
    if (_comm_list == NULL)  {
        return errh->error("Unable to create the GPU communication list");
    }

    for (int th_id = 0; th_id < _workers.size(); th_id++) {
        if (!_workers[th_id]) continue;
        if (_verbose) click_chatter("%p{element} Initializing worker thread %d", this, th_id);

        state &s = _state.get_value(th_id);

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);

        s.timer = new Timer(this);
        s.timer->initialize(this);
        s.timer->move_thread(th_id);
        s.timer->schedule_now();

        s.ring.initialize(_capacity);
    }

    if (_verbose) click_chatter("%p{element} Initializing master thread (%d)", this, _master);
    state &s = _state.get_value_for_thread(_master);
    s.task = new Task(this);
    ScheduleInfo::initialize_task(this, s.task, false, errh);
    s.task->move_thread(_master);
    s.timer = new Timer(this);
    s.timer->initialize(this);
    s.timer->move_thread(_master);
    s.timer->schedule_now();

    // wrapper_ether_mirror_persistent(_comm_list, _comm_list_size, 1, 1024, NULL);

    return 0;
}

#if HAVE_BATCH

/* Sends the batch to the GPU */
void GPUEtherMirrorMW::push_batch(int port, PacketBatch *batch) {
    ErrorHandler *errh = ErrorHandler::default_handler();
    enum rte_gpu_comm_list_status status;
    int ret;
    unsigned n = batch->count();
    rte_mbuf *array[n];

    int i = 0;
    FOR_EACH_PACKET(batch, p)
        array[i++] = p->mb();
    
    do {
        ret = rte_gpu_comm_get_status(_comm_list + _comm_list_put_index, &status);
        if (ret != 0) {
            errh->error("rte_gpu_comm_get_status returned error %d\n", ret);
            return;
        }

        if (status != RTE_GPU_COMM_LIST_FREE) {
            if (_block) {
                _state->task->reschedule();
                if (unlikely(_verbose))
                    click_chatter("%p{element} Congestion: List is not free!" 
                        "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this);
            } else {
                batch->kill();
                if (unlikely(_verbose))
                    click_chatter("%p{element} Dropped %d packets: List is not free!" 
                        "GPU processing is likely too slow, or the list is too small, consider increasing CAPACITY", this, n);

                return;
            }
        }
    } while(unlikely(status != RTE_GPU_COMM_LIST_FREE));
    
    ret = rte_gpu_comm_populate_list_pkts(_comm_list + _comm_list_put_index, array, n);     // includes call to gpu_wmb
    if (ret != 0) {
        errh->error("rte_gpu_comm_populate_list_pkts returned error %d (size was %d )\n", ret, n);
        batch->kill();
        return;
    }
    // rte_wmb();
    
    wrapper_ether_mirror(_comm_list + _comm_list_put_index, n);
    _comm_list_put_index = (_comm_list_put_index + 1) % _comm_list_size;
    _state->task->reschedule();
}

bool GPUEtherMirrorMW::run_task(Task *task) {
    if (click_current_cpu_id() == _master) {
        /* Master */
        enum rte_gpu_comm_list_status status;
        uint32_t n;
        struct rte_mbuf **pkts;

        rte_gpu_comm_get_status(_comm_list + _comm_list_get_index, &status);
        if (status != RTE_GPU_COMM_LIST_DONE) {
            task->fast_reschedule();
            return false;
        }

        rte_rmb();

        /* Batch the processed packets */
        PacketBatch *head = nullptr;
        WritablePacket *packet, *last;
        n = _comm_list[_comm_list_get_index].num_pkts;
        pkts = _comm_list[_comm_list_get_index].mbufs;
        for (uint32_t i = 0; i < n; i++) {
            packet = static_cast<WritablePacket *>(Packet::make(pkts[i], false));   // keep the COLOR annotation!
            if (head == NULL) 
                head = PacketBatch::start_head(packet);
            else
                last->set_next(packet);
            last = packet;
        }
        if (head) {
            head->make_tail(last, n);
            int th_id = PAINT_ANNO(head->first());
            _state.get_value_for_thread(th_id).ring.insert(head->first());
            _state.get_value_for_thread(th_id).task->reschedule();
        }

        rte_gpu_comm_cleanup_list(_comm_list + _comm_list_get_index);
        rte_mb();

        _comm_list_get_index = (_comm_list_get_index + 1) % _comm_list_size;
        return true;

    } else {
        /* Workers */
        PacketBatch* b = reinterpret_cast<PacketBatch*>(_state->ring.extract());
        if (b) output_push_batch(0, b);
        return b != 0;
    }
}

#endif

void GPUEtherMirrorMW::run_timer(Timer *timer) {
    _state->task->reschedule();
    timer->schedule_now();
}

bool GPUEtherMirrorMW::get_spawning_threads(Bitvector& bmp, bool isoutput, int port) {
    auto workers = _pipeliner->get_pushing_threads();
    bmp.clear();
    for (int th_id = 0; th_id < workers.size(); th_id++)
        if (workers[th_id])
            bmp[th_id] = 1;
    return false;
}

void GPUEtherMirrorMW::cleanup(CleanupStage) {
    ErrorHandler *errh = ErrorHandler::default_handler();

    for (int i = 0; i < _state.weight(); i++) {
        if (!_workers[i] || i == _master) continue;

        state &s = _state.get_value(i);

        /* Finish all the tasks polling on the GPU communication list */
        delete s.task;
        s.task = nullptr;
    }
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUEtherMirrorMW)