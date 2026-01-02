#pragma once

#include "utils.h"

///apelat o data la inceputul fiecarei runde pentru a trimite pachete din hosturi.
struct TrafficPattern {
    std::array<Node, DFLY_SIZE>& dfly;
    std::random_device rd;
    std::mt19937 mt;

    TrafficPattern(std::array<Node, DFLY_SIZE>& dfly): dfly(dfly), mt(DEBUG? 0: rd()) {}

    virtual void step() {}
};

struct GroupIncast: TrafficPattern {
    int group_id_incast;
    std::uniform_int_distribution<int> dist;

    GroupIncast(std::array<Node, DFLY_SIZE>& dfly, int group_id_incast): TrafficPattern(dfly), group_id_incast(group_id_incast),
        dist(group_id_incast * GROUP_SIZE, group_id_incast * GROUP_SIZE + HALF_K * HALF_K - 1) {}

    void step();
};

struct HostIncast: TrafficPattern {
    int host_id_incast;

    HostIncast(std::array<Node, DFLY_SIZE>& dfly, int host_id_incast): TrafficPattern(dfly), host_id_incast(host_id_incast) {}

    void step();
};

struct AllToAll: TrafficPattern {
    std::uniform_int_distribution<int> dist;

    AllToAll(std::array<Node, DFLY_SIZE>& dfly): TrafficPattern(dfly), dist(0, DFLY_SIZE - 1) {}

    void step();
};
