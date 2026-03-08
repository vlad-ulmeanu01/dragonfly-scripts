#include "traffic_patterns.h"

/*
TODO:
PACKS_GEN_PER_STEP, WIRE_TRANS_PER_STEP
permutation
incast + bystanders
allreduce (ring, butterfly)
alltoall (in articol: limit endpoint to n parallel connections)
sim 2% links failed
*/

TrafficPattern::TrafficPattern(DflyPlusMaxHosts& dfly): dfly(dfly), mt(DEBUG? 0: rd()), finished_send(false) {}


GroupIncast::GroupIncast(DflyPlusMaxHosts& dfly, int group_id_incast):
    TrafficPattern(dfly), group_id_incast(group_id_incast),
    dist(group_id_incast * dfly.GROUP_SIZE, group_id_incast * dfly.GROUP_SIZE + dfly.HALF_K * dfly.HALF_K - 1) {}

void GroupIncast::step() {
    for (int group_id = 0; group_id < dfly.CNT_GROUPS; group_id++) {
        if (group_id != group_id_incast) {
            for (int i = group_id * dfly.GROUP_SIZE; i < group_id * dfly.GROUP_SIZE + dfly.HALF_K * dfly.HALF_K; i++) {
                for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                    dfly.topo[i][0].out_qu.emplace(i, dist(mt));
                }
            }
        }
    }
}


HostIncast::HostIncast(DflyPlusMaxHosts& dfly, int host_id_incast): TrafficPattern(dfly), host_id_incast(host_id_incast) {}

void HostIncast::step() {
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (i != host_id_incast && dfly.is_node_host(i)) {
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[i][0].out_qu.emplace(i, host_id_incast);
            }
        }
    }
}


AllToAllRing::AllToAllRing(DflyPlusMaxHosts& dfly): TrafficPattern(dfly), step_count(0), instep_count(0) {
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (dfly.is_node_host(i)) host_ids.push_back(i);
    }
}

void AllToAllRing::step() {
    if (finished_send) return; /// TODO cf cu finished send.

    for (int i = 0; i < (int)host_ids.size(); i++) {
        int x = host_ids[i], y = host_ids[(i+1) % host_ids.size()];
        for (int _ = 0; _ < std::min(dfly.PACKS_GEN_PER_STEP, dfly.DFLY_SIZE-1 - step_count - instep_count); _++) {
            dfly.topo[x][0].out_qu.emplace(x, y);
        }
    }

    instep_count += dfly.PACKS_GEN_PER_STEP;
    if (instep_count >= dfly.DFLY_SIZE-1 - step_count) {
        instep_count = 0;
        step_count++;
    }

    if (step_count > dfly.DFLY_SIZE - 1) finished_send = true;
}
