#include <click/config.h>
#include "trim.hh"
#include <click/args.hh>
#include <click/error.hh>
#include <click/glue.hh>
CLICK_DECLS

Trim::Trim()
{
}

int
Trim::configure(Vector<String> &conf, ErrorHandler *errh)
{
    return Args(conf, this, errh).read_mp("LENGTH", _nbytes).complete();
}

#if HAVE_BATCH
PacketBatch *
Trim::simple_action_batch(PacketBatch *head)
{
	FOR_EACH_PACKET(head, current) {
		current->take(_nbytes);
	}
	return head;
}
#endif
Packet *
Trim::simple_action(Packet *p)
{
    p->take(_nbytes);
    return p;
}

CLICK_ENDDECLS
EXPORT_ELEMENT(Trim)
ELEMENT_MT_SAFE(Trim)
