#pragma once

#include "utils.h"


///apelat o data la inceputul fiecarei runde pentru a trimite pachete din hosturi.
struct TrafficPattern {
    DflyPlusMaxHosts& dfly;
    std::random_device rd;
    std::mt19937 mt;

    bool finished_send;
    int cnt_sent_packets, cnt_delivered_packets;
    int step_id;

    TrafficPattern(DflyPlusMaxHosts& dfly);
    virtual ~TrafficPattern() = default;

    virtual void step() = 0;
    virtual Stats get_stats() = 0;

    std::vector<int> get_cnt_packs_held_by_group(bool use_last_tstep);
};

struct HostIncast: TrafficPattern {
    int host_id_incast;

    HostIncast(DflyPlusMaxHosts& dfly, int host_id_incast);

    void step();
    Stats get_stats();
};

struct Permutation: TrafficPattern {
    std::vector<int> perm;

    Permutation(DflyPlusMaxHosts& dfly);

    void step();
    Stats get_stats();
};

///ultimele cnt_groups_incasting fac incast. restul fac permutation traffic.
///ne intereseaza statistici pe traficul permutare.
struct HostIncastWithPermutation: TrafficPattern {
    int host_id_incast;
    int cnt_groups_incasting;
    std::vector<int> perm;

    HostIncastWithPermutation(DflyPlusMaxHosts& dfly, int host_id_incast, int cnt_groups_incasting);

    void step();
    Stats get_stats();
};

///identic cu AllToAllRing la comunicare.
struct AllReduceRing: TrafficPattern {
    int sync_step;
    std::vector<int> host_ids;

    AllReduceRing(DflyPlusMaxHosts& dfly);

    void step();
    Stats get_stats();
};

struct AllReduceButterfly: TrafficPattern {
    int sync_step, pas, max_pas;
    std::vector<int> host_ids;

    AllReduceButterfly(DflyPlusMaxHosts& dfly);

    void step();
    Stats get_stats();
};
