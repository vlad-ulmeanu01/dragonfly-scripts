#include "traffic_patterns.h"

/*
TODO:
PACKS_GEN_PER_STEP, WIRE_TRANS_PER_STEP (ok)
permutation (ok)
incast + bystanders (ok)
allreduce (ring, butterfly) (ok)
alltoall (in articol: limit endpoint to n parallel connections) -- (ring echivalent ca traffic pattern cu ring allreduce)
sim 2% links failed
*/



TrafficPattern::TrafficPattern(DflyPlusMaxHosts& dfly): dfly(dfly), mt(DEBUG? 0: rd()), finished_send(false), cnt_sent_packets(0), cnt_delivered_packets(0), step_id(0) {}

std::vector<int> TrafficPattern::get_cnt_packs_held_by_group(bool use_last_tstep) {
    int tstep = step_id - 1;
    if (!use_last_tstep) {
        ///folosesc tstep-ul din end_step_out_qu_sizes pentru care am #maxim de pachete numarate.
        int max_packs_held = -1;
        for (int t = 0; t < step_id; t++) {
            int curr = 0;
            for (int i = 0; i < dfly.DFLY_SIZE; i++) {
                for (const NeighInfo& ni: dfly.topo[i]) {
                    curr += ni.end_step_out_qu_sizes[tstep];
                }
            }

            if (curr >= max_packs_held) {
                max_packs_held = curr;
                tstep = t;
            }
        }
    }

    std::vector<int> packs_held_by_group(dfly.CNT_GROUPS);
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        for (const NeighInfo& ni: dfly.topo[i]) {
            packs_held_by_group[i / dfly.GROUP_SIZE] += ni.end_step_out_qu_sizes[tstep];
        }
    }

    return packs_held_by_group;
}


///------------


HostIncast::HostIncast(DflyPlusMaxHosts& dfly, int host_id_incast): TrafficPattern(dfly), host_id_incast(host_id_incast) {}

void HostIncast::step() {
    step_id++;

    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (i != host_id_incast && dfly.is_node_host(i)) {
            cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[i][0].out_qu.emplace(i, host_id_incast);
            }
        }
    }
}

Stats HostIncast::get_stats() {
    std::vector<int> phg = get_cnt_packs_held_by_group(true);
    phg.erase(phg.begin() + host_id_incast / dfly.GROUP_SIZE);
    return Stats(phg);
}


///------------


Permutation::Permutation(DflyPlusMaxHosts& dfly): TrafficPattern(dfly), perm(dfly.CNT_GROUPS) {
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), mt);
}

void Permutation::step() {
    step_id++;

    ///grupul i trimite trafic lui perm[i].
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (dfly.is_node_host(i)) {
            int group_to = perm[i / dfly.GROUP_SIZE], l = group_to * dfly.GROUP_SIZE, r = l + dfly.HALF_K * dfly.HALF_K - 1;

            cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[i][0].out_qu.emplace(i, std::uniform_int_distribution<int>(l, r)(mt));
            }
        }
    }
}

Stats Permutation::get_stats() {
    std::vector<int> phg = get_cnt_packs_held_by_group(true);
    return Stats(phg);
}


///------------


HostIncastWithPermutation::HostIncastWithPermutation(DflyPlusMaxHosts& dfly, int host_id_incast, int cnt_groups_incasting):
    TrafficPattern(dfly), host_id_incast(host_id_incast), cnt_groups_incasting(cnt_groups_incasting),
    perm(dfly.CNT_GROUPS - cnt_groups_incasting)
{
    assert(host_id_incast / dfly.GROUP_SIZE >= (int)perm.size());

    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), mt);
}

void HostIncastWithPermutation::step() {
    step_id++;

    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (dfly.is_node_host(i)) {
            int group_from = i / dfly.GROUP_SIZE;

            if (group_from < (int)perm.size()) {
                int group_to = perm[group_from], l = group_to * dfly.GROUP_SIZE, r = l + dfly.HALF_K * dfly.HALF_K - 1;

                cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
                for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                    dfly.topo[i][0].out_qu.emplace(i, std::uniform_int_distribution<int>(l, r)(mt));
                }
            } else if (i != host_id_incast && dfly.is_node_host(i)) {
                cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
                for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                    dfly.topo[i][0].out_qu.emplace(i, host_id_incast);
                }
            }
        }
    }
}

Stats HostIncastWithPermutation::get_stats() {
    std::vector<int> phg = get_cnt_packs_held_by_group(true);
    phg.erase(phg.begin() + perm.size(), phg.end());
    return Stats(phg);
}


///------------


AllReduceRing::AllReduceRing(DflyPlusMaxHosts& dfly): TrafficPattern(dfly), sync_step(0) {
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (dfly.is_node_host(i)) host_ids.push_back(i);
    }
}

void AllReduceRing::step() {
    step_id++;

    ///astept sa ajunga toate pachetele din pasul curent inainte sa trimit altele.
    if (cnt_sent_packets > cnt_delivered_packets || finished_send) return;

    for (int i = 0; i < (int)host_ids.size(); i++) {
        int x = host_ids[i], y = host_ids[(i+1) % host_ids.size()];

        cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
        for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
            dfly.topo[x][0].out_qu.emplace(x, y);
        }
    }

    sync_step++;
    if (sync_step >= (int)host_ids.size() - 1) finished_send = true;
}

Stats AllReduceRing::get_stats() {
    std::vector<int> phg = get_cnt_packs_held_by_group(false);
    return Stats(phg);
}


///------------

AllReduceButterfly::AllReduceButterfly(DflyPlusMaxHosts& dfly): TrafficPattern(dfly), sync_step(0), pas(1), max_pas(1) {
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        if (dfly.is_node_host(i)) host_ids.push_back(i);
    }

    while ((max_pas << 1) <= (int)host_ids.size()) max_pas <<= 1;
}

void AllReduceButterfly::step() {
    step_id++;

    ///astept sa ajunga toate pachetele din pasul curent inainte sa trimit altele.
    if (cnt_sent_packets > cnt_delivered_packets || finished_send) return;

    if (sync_step == 0) {
        ///transfer tot ce este peste max_pas in partea putere de 2:
        for (int i = max_pas; i < (int)host_ids.size(); i++) {
            cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[host_ids[i]][0].out_qu.emplace(host_ids[i], host_ids[i - max_pas]);
            }
        }
    } else if (pas < max_pas) {
        for (int i = 0; i < max_pas; i++) {
            cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[host_ids[i]][0].out_qu.emplace(host_ids[i], host_ids[i ^ pas]);
            }
        }
        pas <<= 1;
    } else {
        /// transfer inapoi in ciot reducerea construita.
        for (int i = max_pas; i < (int)host_ids.size(); i++) {
            cnt_sent_packets += dfly.PACKS_GEN_PER_STEP;
            for (int _ = 0; _ < dfly.PACKS_GEN_PER_STEP; _++) {
                dfly.topo[host_ids[i]][0].out_qu.emplace(host_ids[i - max_pas], host_ids[i]);
            }
        }
        
        finished_send = true;
    }

    sync_step++;
}

Stats AllReduceButterfly::get_stats() {
    std::vector<int> phg = get_cnt_packs_held_by_group(false);
    return Stats(phg);
}
