#ifndef ROUTE_H
#define ROUTE_H

struct Route {
    IPAddress addr;
    IPAddress mask;
    IPAddress gw;
    int32_t port;
    int32_t extra;
};

#endif