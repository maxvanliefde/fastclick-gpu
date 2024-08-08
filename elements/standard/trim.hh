#ifndef CLICK_TRIM_HH
#define CLICK_TRIM_HH
#include <click/batchelement.hh>
CLICK_DECLS

/*
 * =c
 * Trim(LENGTH)
 * =s basicmod
 * trims bytes from back of packets
 * =d
 * Deletes the last LENGTH bytes from each packet.
 * =e
 * Use this to get rid of the CRC field:
 *
 *   Trim(4)
 */

class Trim : public BatchElement { public:

    Trim() CLICK_COLD;

    const char *class_name() const override		{ return "Trim"; }
    const char *port_count() const override		{ return PORTS_1_1; }

    int configure(Vector<String> &, ErrorHandler *) CLICK_COLD;

#if HAVE_BATCH
    PacketBatch *simple_action_batch(PacketBatch *);
#endif
  Packet *simple_action(Packet *);

  private:

    unsigned _nbytes;

};

CLICK_ENDDECLS
#endif
