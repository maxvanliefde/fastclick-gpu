// -*- c-basic-offset: 4 -*-
/*
 * lineariplookup.{cc,hh} -- element looks up next-hop address in linear
 * routing table
 * Robert Morris, Eddie Kohler
 *
 * Copyright (c) 1999-2000 Massachusetts Institute of Technology
 * Copyright (c) 2002 International Computer Science Institute
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>
#include "lineariplookup.hh"
#include <click/ipaddress.hh>
#include <click/straccum.hh>
#include <click/error.hh>
#include <click/args.hh>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
CLICK_DECLS

LinearIPLookup::LinearIPLookup()
    : _zero_route(-1)
{
}

LinearIPLookup::~LinearIPLookup()
{
}

int
LinearIPLookup::initialize(ErrorHandler *)
{
    _last_addr = IPAddress();
#ifdef IP_RT_CACHE2
    _last_addr2 = _last_addr;
#endif
    return 0;
}

int LinearIPLookup::configure(Vector<String> &conf, ErrorHandler *errh)
{
    if (Args(conf, this, errh)
    .read("LOOKUP_TABLE", _lookup_table)
        .consume() < 0)
    {
        return -1;
    }

    printf("table: %d\n", _lookup_table);

    read_from_file(_lookup_table);
    printf("size: %d\n", _t.size());
    return 0;
}

int LinearIPLookup::save_to_file(uint64_t size) {
    std::string file_name = "saved_vector" + std::to_string(size) + ".bin";
    
    ofstream fout(file_name, ios::out | ios::binary);
    fout<<_t.size()<<endl;
    for (int i = 0; i < _t.size(); i++) {
        fout<<_t[i].addr<<endl;
        fout<<_t[i].mask<<endl;
        fout<<_t[i].gw<<endl;
        fout<<_t[i].port<<endl;
        fout<<_t[i].extra<<endl;
    }
    fout.close();
	return 0;
}

int LinearIPLookup::read_from_file(uint8_t table) {
    std::string file_name;

    switch(table) {
        case 0:
            file_name = "../saved_vector100.bin";
            break;
        case 1:
            file_name = "../saved_vector1000.bin";
            break;
        case 2:
            file_name = "../saved_vector10000.bin";
            break;
        case 3:
            file_name = "../saved_vector50000.bin";
            break;
        case 4:
            file_name = "../saved_vector100000.bin";
            break;
        case 5:
            file_name = "../saved_vector1000000.bin";
            break;
        default:
            file_name = "../saved_vector100.bin";
            break;
    }
    
    std::ifstream fin(file_name, std::ios::in | std::ios::binary);
    std::string line;
    std:getline(fin, line);
    uint32_t size = std::stoul(line);
    _t.resize(size);
    for(int i = 0; i < size; i++) {
        std::getline(fin, line);
        uint32_t addr = std::stoul(line);
        _t[i].addr = addr;

        std::getline(fin, line);
        uint32_t mask = std::stoul(line);
        _t[i].mask = mask;

        std::getline(fin, line);
        uint32_t gw = std::stoul(line);
        _t[i].gw = gw;

        std::getline(fin, line);
        uint32_t port = std::stoul(line);
        _t[i].port = port;

        std::getline(fin, line);
        uint32_t extra = std::stoul(line);
        _t[i].extra = extra;

    }
    fin.close();

    return 0;

}

bool
LinearIPLookup::check() const
{
    bool ok = true;
    //click_chatter("%s\n", ((LinearIPLookup*)this)->dump_routes().c_str());

    // 'next' pointers are correct
    for (int i = 0; i < _t.size(); i++) {
	if (!_t[i].real())
	    continue;
	for (int j = i + 1; j < _t[i].extra && j < _t.size(); j++)
	    if (_t[i].contains(_t[j]) && _t[j].real()) {
		click_chatter("%s: bad next pointers: routes %s, %s", declaration().c_str(), _t[i].unparse_addr().c_str(), _t[j].unparse_addr().c_str());
		ok = false;
	    }
#if 0
	// This invariant actually does not hold.
	int j = _t[i].extra;
	if (j < _t.size())
	    if (!_t[i].contains(_t[j]) && _t[j].real()) {
		click_chatter("%s: bad next pointers': routes %s, %s", declaration().c_str(), _t[i].unparse_addr().c_str(), _t[j].unparse_addr().c_str());
		ok = false;
	    }
#endif
    }

    // no duplicate routes
    for (int i = 0; i < _t.size(); i++)
	for (int j = i + 1; j < _t.size(); j++)
	    if (_t[i].addr == _t[j].addr && _t[i].mask == _t[j].mask && _t[i].real() && _t[j].real()) {
		click_chatter("%s: duplicate routes for %s", declaration().c_str(), _t[i].unparse_addr().c_str());
		ok = false;
	    }

    // caches point to the right place
    if (_last_addr && lookup_entry(_last_addr) != _last_entry) {
	click_chatter("%s: bad cache entry for %s", declaration().c_str(), _last_addr.unparse().c_str());
	ok = false;
    }
#ifdef IP_RT_CACHE2
    if (_last_addr2 && lookup_entry(_last_addr2) != _last_entry2) {
	click_chatter("%s: bad cache entry for %s", declaration().c_str(), _last_addr2.unparse().c_str());
	ok = false;
    }
#endif

    return ok;
}

int
LinearIPLookup::add_route(const IPRoute &r, bool allow_replace, IPRoute* replaced, ErrorHandler *)
{
    // overwrite any existing route
    int found = _t.size();
    for (int i = 0; i < _t.size(); i++)
	if (!_t[i].real()) {
	    if (found == _t.size())
		found = i;
	} else if (_t[i].addr == r.addr && _t[i].mask == r.mask) {
	    if (replaced)
		*replaced = _t[i];
	    if (!allow_replace)
		return -EEXIST;
	    _t[i].gw = r.gw;
	    _t[i].port = r.port;
	    // check();
	    return 0;
	}

    // maybe make a new slot
    if (found == _t.size())
	_t.push_back(r);
    else
	_t[found] = r;

    // patch up next pointers
    _t[found].extra = 0x7FFFFFFF;
    for (int i = found - 1; i >= 0; i--)
	if (_t[i].contains(r) && _t[i].extra > found)
	    _t[i].extra = found;
    for (int i = found + 1; i < _t.size(); i++)
	if (r.contains(_t[i]) && _t[i].real()) {
	    _t[found].extra = i;
	    break;
	}

    // remember zero route
    if (!r.addr && r.mask.addr() == 0xFFFFFFFFU)
	_zero_route = found;

    // get rid of caches
    _last_addr = IPAddress();
#ifdef IP_RT_CACHE2
    _last_addr2 = IPAddress();
#endif

    // check();

    // Saving the structure into files to load it faster later
    // if (_t.size() == 100) {
    //     save_to_file(100);
    // }
    // if (_t.size() == 1000) {
    //     save_to_file(1000);
    // }
    // if (_t.size() == 10000) {
    //     save_to_file(10000);
    // }
    // if (_t.size() == 50000) {
    //     save_to_file(50000);
    // }
    // if (_t.size() == 100000) {
    //     save_to_file(100000);
    // }
    // if (_t.size() == 1000000) {
    //     save_to_file(1000000);
    // }

    return 0;
}

int
LinearIPLookup::remove_route(const IPRoute& route, IPRoute* old_route, ErrorHandler *errh)
{
    for (int i = 0; i < _t.size(); i++)
	if (route.match(_t[i])) {
	    if (old_route)
		*old_route = _t[i];
	    _t[i].kill();

	    // need to handle zero routes, bummer
	    if (i == _zero_route)
		_zero_route = -1;
	    else if (i < _zero_route) {
		IPRoute zero(_t[_zero_route]);
		_t[_zero_route].kill();
		int r = add_route(zero, false, 0, errh);
		assert(r >= 0);
		(void) r;
	    }

	    // get rid of caches
	    _last_addr = IPAddress();
#ifdef IP_RT_CACHE2
	    _last_addr2 = IPAddress();
#endif
	    // check();
	    return 0;
	}
    return -ENOENT;
}

int
LinearIPLookup::lookup_entry(IPAddress a) const
{
    for (int i = 0; i < _t.size(); i++){
	if (_t[i].contains(a)) {
	    int found = i;
	    for (int j = _t[i].extra; j < _t.size(); j++)
		if (_t[j].contains(a) && _t[j].mask_as_specific(_t[found].mask))
		    found = j;
	    return found;
	}
    }
    return -1;
}

int
LinearIPLookup::lookup_route(IPAddress a, IPAddress &gw) const
{
    int ei = lookup_entry(a);
    if (ei >= 0) {
	gw = _t[ei].gw;
	return _t[ei].port;
    } else
	return -1;
}

String
LinearIPLookup::dump_routes()
{
    StringAccum sa;
    for (int i = 0; i < _t.size(); i++)
	if (_t[i].real())
	    _t[i].unparse(sa, true) << '\n';
    return sa.take_string();
}

inline int
LinearIPLookup::smaction(Packet* p) {
#define EXCHANGE(a,b,t) { t = a; a = b; b = t; }
    IPAddress a = p->dst_ip_anno();
    int ei = -1;

    if (a && a == _last_addr)
    ei = _last_entry;
#ifdef IP_RT_CACHE2
    else if (a && a == _last_addr2)
    ei = _last_entry2;
#endif
    else if ((ei = lookup_entry(a)) >= 0) {
#ifdef IP_RT_CACHE2
    _last_addr2 = _last_addr;
    _last_entry2 = _last_entry;
#endif
    _last_addr = a;
    _last_entry = ei;
    } else {
    static int complained = 0;
    if (++complained <= 5)
        click_chatter("LinearIPLookup: no route for %s", a.unparse().c_str());
    return -1;
    }

    const IPRoute &e = _t[ei];
    if (e.gw)
    p->set_dst_ip_anno(e.gw);
    return e.port;
}

void
LinearIPLookup::push(int, Packet *p)
{
    int out = smaction(p);
    if (unlikely(out == -1)) {
        p->kill();
        return;
    }
    output(out).push(p);
}

#if HAVE_BATCH
void LinearIPLookup::push_batch(int, PacketBatch *batch) {
    CLASSIFY_EACH_PACKET(noutputs() + 1,smaction,batch,checked_output_push_batch);
}
#endif

CLICK_ENDDECLS
ELEMENT_REQUIRES(IPRouteTable)
EXPORT_ELEMENT(LinearIPLookup)