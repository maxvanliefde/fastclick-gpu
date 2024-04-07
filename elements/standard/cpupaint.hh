#ifndef CLICK_CPUPAINT_HH
#define CLICK_CPUPAINT_HH
#include <click/batchelement.hh>
CLICK_DECLS

/*
=c

CPUPaint(COLOR [, ANNO])

=s paint

sets packet paint annotations

=d

Sets each packet's paint annotation to COLOR, an integer 0..255.

CPUPaint sets the packet's PAINT annotation by default, but the ANNO argument can
specify any one-byte annotation.

=h color read/write

Get/set the color to paint.

=a PaintTee */

class CPUPaint : public BatchElement { public:

    CPUPaint() CLICK_COLD;

    const char *class_name() const override		{ return "CPUPaint"; }
    const char *port_count() const override		{ return PORTS_1_1; }

    int configure(Vector<String> &, ErrorHandler *) CLICK_COLD;
    bool can_live_reconfigure() const		{ return true; }
    void add_handlers() CLICK_COLD;

#if HAVE_BATCH
    PacketBatch *simple_action_batch(PacketBatch *);
#endif
    Packet *simple_action(Packet *);

  private:

    uint8_t _anno;
    uint8_t _color;

};

CLICK_ENDDECLS
#endif
