#ifndef CLICK_CHECKCRC32_HH
#define CLICK_CHECKCRC32_HH
#include <click/batchelement.hh>
#include <click/atomic.hh>
CLICK_DECLS

/*
 * =c
 * CheckCRC32()
 * =s crc
 * checks packet CRC32s
 * =d
 * Check that the CRC32 appended by SetCRC32 is OK.
 * If so, delete the CRC from the packet.
 * Otherwise, drop the packet.
 * =a SetCRC32
 */
class CheckCRC32 : public SimpleElement<CheckCRC32> { public:

    CheckCRC32();

    const char *class_name() const override		{ return "CheckCRC32"; }
    const char *port_count() const override		{ return PORTS_1_1; }

    Packet *simple_action(Packet *);

  private:

    atomic_uint32_t _drops;

};

CLICK_ENDDECLS
#endif
