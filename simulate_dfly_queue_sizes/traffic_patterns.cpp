#include "traffic_patterns.h"

void GroupIncast::step() {
    for (int group_id = 0; group_id < CNT_GROUPS; group_id++) {
        if (group_id != group_id_incast) {
            for (int i = group_id * GROUP_SIZE; i < group_id * GROUP_SIZE + HALF_K * HALF_K; i++) {
                dfly[i].neighs[0].out_qu.emplace(i, dist(mt));
            }
        }
    }
}

void HostIncast::step() {
    for (int i = 0; i < DFLY_SIZE; i++) {
        if (i != host_id_incast && is_node_host(i)) dfly[i].neighs[0].out_qu.emplace(i, host_id_incast);
    }
}

void AllToAll::step() {
    for (int i = 0; i < DFLY_SIZE; i++) {
        if (is_node_host(i)) {
            int j = dist(mt);
            while (j == i || !is_node_host(j)) j = dist(mt);
            dfly[i].neighs[0].out_qu.emplace(i, j);
        }
    }
}
