#ifndef ROUTEGPU_H
#define ROUTEGPU_H

struct RouteGPU {
    int32_t addr;
    int32_t mask;
    int32_t gw;
    int32_t port;
    int32_t extra;
};

#endif

