#pragma once

#include "utils.h"

///apelat o data la inceputul fiecarei runde pentru a trimite pachete din hosturi.
struct TrafficPattern {
    DflyPlusMaxHosts& dfly;
    std::random_device rd;
    std::mt19937 mt;
    bool finished_send;

    TrafficPattern(DflyPlusMaxHosts& dfly);

    virtual void step() {}
};

struct GroupIncast: TrafficPattern {
    int group_id_incast;
    std::uniform_int_distribution<int> dist;

    GroupIncast(DflyPlusMaxHosts& dfly, int group_id_incast);

    void step();
};

struct HostIncast: TrafficPattern {
    int host_id_incast;

    HostIncast(DflyPlusMaxHosts& dfly, int host_id_incast);

    void step();
};

struct AllToAllRing: TrafficPattern {
    int step_count, instep_count;
    std::vector<int> host_ids;

    AllToAllRing(DflyPlusMaxHosts& dfly);

    void step();
};
