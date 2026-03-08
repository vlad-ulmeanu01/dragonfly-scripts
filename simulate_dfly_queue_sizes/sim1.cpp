#include "utils.h"
#include "traffic_patterns.h"
#include "balls_bins.h"

void step_propagate_packets(DflyPlusMaxHosts& dfly, BallsBins *bb, int& cnt_delivered_packets) {
    std::vector<std::vector<Packet>> inbound_packets(dfly.DFLY_SIZE);

    for (int ind = 0; ind < dfly.DFLY_SIZE; ind++) {
        for (NeighInfo& ni: dfly.topo[ind]) {
            for (int _ = 0; !ni.out_qu.empty() && _ < dfly.WIRE_TRANS_PER_STEP; _++) {
                inbound_packets[ni.id].push_back(ni.out_qu.front());
                ni.out_qu.pop();
            }
        }
    }

    for (int ind = 0; ind < dfly.DFLY_SIZE; ind++) {
        int group_now = ind / dfly.GROUP_SIZE;
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>> need_random;

        for (auto [from, to]: inbound_packets[ind]) {
            if (ind == to) {
                cnt_delivered_packets++;
                continue;
            }

            int group_from = from / dfly.GROUP_SIZE, group_to = to / dfly.GROUP_SIZE;

            if (group_now == group_from) {
                if (dfly.is_node_host(ind)) { ///(det) host -> leaf. (-- la cum e scris acum codul, nu ar tb sa intre niciodata aici)
                    dfly.topo[ind][0].out_qu.emplace(from, to);
                } else if (dfly.is_switch_leaf(ind)) { ///comportament diferit daca group_from == group_to sau nu.
                    auto it = std::find_if(dfly.topo[ind].begin(), dfly.topo[ind].end(), [&to](const NeighInfo& ni) { return ni.id == to; } );

                    if (group_from != group_to || it == dfly.topo[ind].end()) { ///(rand) leaf -> spine.
                        need_random.emplace_back(Packet(from, to), dfly.topo[ind].begin() + dfly.HALF_K, dfly.topo[ind].begin() + dfly.K);
                    } else { ///(det) (acelasi grup) leaf -> host.
                        it->out_qu.emplace(from, to);
                    }
                } else {
                    if (group_from != group_to) { ///(rand) spine -> spine, o sa parasesc grupul actual la urmatorul step (aici doar aleg in care out queue sa-l pun).
                        need_random.emplace_back(Packet(from, to), dfly.topo[ind].begin() + dfly.HALF_K, dfly.topo[ind].begin() + dfly.K);
                    } else { ///(det) (acelasi grup) spine -> leaf.
                        auto it = std::find_if(
                            dfly.topo[ind].begin(), dfly.topo[ind].end(),
                            [&dfly, &to](const NeighInfo& ni) {
                                return std::find_if(
                                    dfly.topo[ni.id].begin(), dfly.topo[ni.id].end(),
                                    [&to](const NeighInfo& ni_oth) { return ni_oth.id == to; }
                                ) != dfly.topo[ni.id].end();
                            }
                        );
                        
                        assert(it != dfly.topo[ind].end());
                        it->out_qu.emplace(from, to);
                    }
                }
            } else if (group_now == group_to) {
                if (!dfly.is_switch_leaf(ind)) { ///(det) spine -> leaf.
                    auto it = std::find_if(
                        dfly.topo[ind].begin(), dfly.topo[ind].end(),
                        [&dfly, &to](const NeighInfo& ni) {
                            return std::find_if(
                                dfly.topo[ni.id].begin(), dfly.topo[ni.id].end(),
                                [&to](const NeighInfo& ni_oth) { return ni_oth.id == to; }
                            ) != dfly.topo[ni.id].end();
                        }
                    );
                    
                    assert(it != dfly.topo[ind].end());
                    it->out_qu.emplace(from, to);
                } else { ///(det) leaf -> host (== to).
                    auto it = std::find_if(dfly.topo[ind].begin(), dfly.topo[ind].end(), [&to](const NeighInfo& ni) { return ni.id == to; } );
                    assert(it != dfly.topo[ind].end());
                    it->out_qu.emplace(from, to);
                }
            } else { ///sunt in spine-ul intermediar.
                assert(!dfly.is_node_host(ind));

                if (!dfly.is_switch_leaf(ind)) { ///spine.
                    auto it = std::find_if(
                        dfly.topo[ind].begin(), dfly.topo[ind].end(),
                        [&dfly, &group_to](const NeighInfo& ni) { return ni.id / dfly.GROUP_SIZE == group_to; }
                    );

                    if (it != dfly.topo[ind].end()) { ///(det) nu trebuie sa schimb spine-ul, am legatura directa catre group_to.
                        it->out_qu.emplace(from, to);
                    } else { ///(rand) spine -> leaf.
                        need_random.emplace_back(Packet(from, to), dfly.topo[ind].begin(), dfly.topo[ind].begin() + dfly.HALF_K);
                    }
                } else { ///(det) leaf -> spine.
                    auto it = std::find_if(
                        dfly.topo[ind].begin(), dfly.topo[ind].end(),
                        [&group_to, &dfly](const NeighInfo& ni) {
                            return std::find_if(
                                dfly.topo[ni.id].begin(), dfly.topo[ni.id].end(),
                                [&dfly, &group_to](const NeighInfo& ni_oth) { return ni_oth.id / dfly.GROUP_SIZE == group_to; }
                            ) != dfly.topo[ni.id].end();
                        }
                    );

                    assert(it != dfly.topo[ind].end());
                    it->out_qu.emplace(from, to);
                }
            }
        }

        bb->make_picks(need_random);
    }
}

void end_step(DflyPlusMaxHosts& dfly) {
    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        for (NeighInfo& ni: dfly.topo[i]) {
            ni.end_step_qu_sizes.push_back(ni.out_qu.size());
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "Usage: 1) K\n";
        std::cerr << "       2) <packets generated per step per host> (PACKS_GEN_PER_STEP)\n";
        std::cerr << "       3) <how many packets can a connection transmit per step> (WIRE_TRANS_PER_STEP, should be <= HALF_K**2)\n";
        std::cerr << "       4) <global topology configuration (random, configs/k_4/config_k_4_score_0.txt)>\n";
        std::cerr << "       5) <traffic pattern> (group_incast, host_incast, all_to_all_ring)\n";
        std::cerr << "       6) <balls bins strategy> (greedy1, greedy2seq, greedy2par)\n";
        std::cerr << "       7) <cnt steps> (any integer >= 1)\n";
        std::cerr << "ex ./sim1 4 1 4 random group_incast greedy1 100\n";
        return 0;
    }

    int K = atoi(argv[1]), PACKS_GEN_PER_STEP = atoi(argv[2]), WIRE_TRANS_PER_STEP = atoi(argv[3]);
    std::string cfg_type(argv[4]), traffic_pattern(argv[5]), balls_bins(argv[6]);
    int cnt_steps = atoi(argv[7]);

    DflyPlusMaxHosts dfly(K, PACKS_GEN_PER_STEP, WIRE_TRANS_PER_STEP, cfg_type);

    // dfly.dbg_topo();

    // auto spine_cfg = dfly.dbg_get_spine_cfg();
    // dbgs2d(spine_cfg);

    // int score = dfly.get_score();
    // dbgln(score);

    std::unique_ptr<TrafficPattern> tp;
    if (traffic_pattern == "group_incast") tp = std::make_unique<GroupIncast>(dfly, 0);
    else if (traffic_pattern == "host_incast") tp = std::make_unique<HostIncast>(dfly, 0);
    else if (traffic_pattern == "all_to_all_ring") tp = std::make_unique<AllToAllRing>(dfly);
    else assert(false);

    std::unique_ptr<BallsBins> bb;
    if (balls_bins == "greedy1") bb = std::make_unique<Greedy1>(dfly);
    else if (balls_bins == "greedy2seq") bb = std::make_unique<Greedy2seq>(dfly);
    else if (balls_bins == "greedy2par") bb = std::make_unique<Greedy2par>(dfly);
    else assert(false);

    int cnt_delivered_packets = 0;
    for (int step_id = 0; step_id < cnt_steps; step_id++) {
        tp->step();
        step_propagate_packets(dfly, bb.get(), cnt_delivered_packets);
        end_step(dfly);
    }

    for (int i = 0; i < dfly.DFLY_SIZE; i++) {
        std::cout << i << ' ' << dfly.topo[i].size() << '\n';
        for (const NeighInfo& ni: dfly.topo[i]) std::cout << ni.end_step_qu_sizes.back() << ' ';
        std::cout << '\n';
    }

    // std::cout << "cnt_delivered_packets = " << cnt_delivered_packets << '\n';

    return 0;
}
