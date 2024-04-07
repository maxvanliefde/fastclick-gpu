#include <click/config.h>
#include "cpupaint.hh"
#include <click/args.hh>
#include <click/error.hh>
#include <click/glue.hh>
#include <click/packet_anno.hh>
CLICK_DECLS

CPUPaint::CPUPaint()
{
}

int
CPUPaint::configure(Vector<String> &conf, ErrorHandler *errh)
{
    int anno = PAINT_ANNO_OFFSET;
    if (Args(conf, this, errh)
	.read_p("ANNO", AnnoArg(1), anno).complete() < 0)
	return -1;
    _anno = anno;
    return 0;
}

#if HAVE_BATCH
PacketBatch *
CPUPaint::simple_action_batch(PacketBatch *batch)
{
    // printf("received on %d\n", click_current_cpu_id());
    FOR_EACH_PACKET(batch, cur) {	
		cur->set_anno_u8(_anno, click_current_cpu_id());
	}
    return batch;
}
#endif
Packet *
CPUPaint::simple_action(Packet *p)
{
    p->set_anno_u8(_anno, click_current_cpu_id());
    return p;
}

void
CPUPaint::add_handlers()
{
    add_data_handlers("color", Handler::OP_READ | Handler::OP_WRITE, &_color);
}

CLICK_ENDDECLS
EXPORT_ELEMENT(CPUPaint)
ELEMENT_MT_SAFE(CPUPaint)
