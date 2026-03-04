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

void GroupIncast::step() {
    for (int group_id = 0; group_id < CNT_GROUPS; group_id++) {
        if (group_id != group_id_incast) {
            for (int i = group_id * GROUP_SIZE; i < group_id * GROUP_SIZE + HALF_K * HALF_K; i++) {
                for (int _ = 0; _ < PACKS_GEN_PER_STEP; _++) {
                    dfly[i].neighs[0].out_qu.emplace(i, dist(mt));
                }
            }
        }
    }
}

void HostIncast::step() {
    for (int i = 0; i < DFLY_SIZE; i++) {
        if (i != host_id_incast && is_node_host(i)) {
            for (int _ = 0; _ < PACKS_GEN_PER_STEP; _++) {
                dfly[i].neighs[0].out_qu.emplace(i, host_id_incast);
            }
        }
    }
}

void AllToAllRing::step() {
    if (finished_send) return;

    for (int i = 0; i < DFLY_SIZE; i++) {
        if (is_node_host(i)) {
            for (int _ = 0; _ < std::min(PACKS_GEN_PER_STEP, DFLY_SIZE-1 - step_count - instep_count); _++) {
                dfly[i].neighs[0].out_qu.emplace(i, (i+1) % DFLY_SIZE);
            }
        }
    }

    instep_count += PACKS_GEN_PER_STEP;
    if (instep_count >= DFLY_SIZE-1 - step_count) {
        instep_count = 0;
        step_count++;
    }

    if (step_count > DFLY_SIZE - 1) finished_send = true;
}
