/*
 * constantbatch.{cc,hh}
 */
#include <click/config.h>
#include <click/glue.hh>
#include <click/args.hh>
#include <click/packet.hh>
#include <click/packet_anno.hh>
#include <click/master.hh>
#include "constantbatch.hh"


CLICK_DECLS

ConstantBatch::ConstantBatch() : _burst(32), _timeout(-1), _verbose(false) {
    in_batch_mode = BATCH_MODE_NEEDED;
}


int
ConstantBatch::configure(Vector<String> &conf, ErrorHandler *errh)
{
    if (Args(conf, this, errh)
        .read_p("BURST", _burst)
        .read_p("TIMER", _timeout)
        .read("VERBOSE", _verbose)
        .complete() < 0)
        return -1;

    if (_timeout >= 0) {
        for (unsigned i = 0; i < click_max_cpu_ids(); i++) {
            State &s = _state.get_value_for_thread(i);
            Task* task = new Task(this);
            task->initialize(this,false);
            task->move_thread(i);
            s.timer = new Timer(task);
            s.timer->initialize(this);
            s.timer->move_thread(i);
        }
    }

    return 0;
}

// only fired if time is out
bool ConstantBatch::run_task(Task *task) {
    State &s = _state.get();

    if (!s.last_batch) {
        return false;
    }

    PacketBatch* p = s.last_batch;
    if (unlikely(_verbose && p->count() != _burst))
        click_chatter("%p{element} Warning: pushing burst of size %d", this, p->count());
    output_push_batch(0,p);
    s.last_batch = nullptr;
    return true;
}

void ConstantBatch::push_batch(int port, PacketBatch *p) {
    // assert(p->count() > 0);
    State &s = _state.get();

    if (s.last_batch == nullptr) {
        s.last_batch = p;
    } else {
        s.last_batch->append_batch(p);
    }

    do {
        if (s.last_batch->count() < _burst) {
            if (_timeout >= 0) {
                s.timer->schedule_after(Timestamp::make_usec(_timeout));
            }
            return;
        } else if (s.last_batch->count() == _burst) {
            if (_timeout >= 0)
                s.timer->unschedule();

            p = s.last_batch;
            s.last_batch = nullptr;
            output_push_batch(0,p);
            return;
        } else {
            if (_timeout >= 0)
                s.timer->unschedule();

            p = s.last_batch;
            PacketBatch* newBatch;
            p->split(_burst, newBatch, true);
            s.last_batch = newBatch;
            output_push_batch(0,p);
        }
    } while(true);
}

CLICK_ENDDECLS
ELEMENT_REQUIRES(batch)
EXPORT_ELEMENT(ConstantBatch)
