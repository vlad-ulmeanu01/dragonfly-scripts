#include "utils.h"
#include "traffic_patterns.h"
#include "balls_bins.h"

void step_propagate_packets(std::array<Node, DFLY_SIZE>& dfly, BallsBins* bb, int& cnt_delivered_packets) {
    std::array<std::vector<Packet>, DFLY_SIZE> inbound_packets = {};
    for (int ind = 0; ind < DFLY_SIZE; ind++) {
        for (NeighInfo& ni: dfly[ind].neighs) {
            if (!ni.out_qu.empty()) {
                inbound_packets[ni.id].push_back(ni.out_qu.front());
                ni.out_qu.pop();
            }
        }
    }

    for (int ind = 0; ind < DFLY_SIZE; ind++) {
        int group_now = ind / GROUP_SIZE;
        std::vector<std::tuple<Packet, std::vector<NeighInfo>::iterator, std::vector<NeighInfo>::iterator>> need_random;

        for (auto [from, to]: inbound_packets[ind]) {
            if (ind == to) {
                cnt_delivered_packets++;
                continue;
            }

            int group_from = from / GROUP_SIZE, group_to = to / GROUP_SIZE;

            if (group_now == group_from) {
                if (is_node_host(ind)) { ///(det) host -> leaf. (-- la cum e scris acum codul, nu ar tb sa intre niciodata aici)
                    dfly[ind].neighs[0].out_qu.emplace(from, to);
                } else if (is_switch_leaf(ind)) { ///comportament diferit daca group_from == group_to sau nu.
                    auto it = std::find_if(dfly[ind].neighs.begin(), dfly[ind].neighs.end(), [&to](const NeighInfo& ni) { return ni.id == to; } );

                    if (group_from != group_to || it == dfly[ind].neighs.end()) { ///(rand) leaf -> spine.
                        need_random.emplace_back(Packet(from, to), dfly[ind].neighs.begin() + HALF_K, dfly[ind].neighs.begin() + K);
                    } else { ///(det) (acelasi grup) leaf -> host.
                        it->out_qu.emplace(from, to);
                    }
                } else {
                    if (group_from != group_to) { ///(rand) spine -> spine, o sa parasesc grupul actual la urmatorul step (aici doar aleg in care out queue sa-l pun).
                        need_random.emplace_back(Packet(from, to), dfly[ind].neighs.begin() + HALF_K, dfly[ind].neighs.begin() + K);
                    } else { ///(det) (acelasi grup) spine -> leaf.
                        auto it = std::find_if(
                            dfly[ind].neighs.begin(), dfly[ind].neighs.end(),
                            [&dfly, &to](const NeighInfo& ni) {
                                return std::find_if(
                                    dfly[ni.id].neighs.begin(), dfly[ni.id].neighs.end(),
                                    [&to](const NeighInfo& ni_oth) { return ni_oth.id == to; }
                                ) != dfly[ni.id].neighs.end();
                            }
                        );
                        
                        assert(it != dfly[ind].neighs.end());
                        it->out_qu.emplace(from, to);
                    }
                }
            } else if (group_now == group_to) {
                if (!is_switch_leaf(ind)) { ///(det) spine -> leaf.
                    auto it = std::find_if(
                        dfly[ind].neighs.begin(), dfly[ind].neighs.end(),
                        [&dfly, &to](const NeighInfo& ni) {
                            return std::find_if(
                                dfly[ni.id].neighs.begin(), dfly[ni.id].neighs.end(),
                                [&to](const NeighInfo& ni_oth) { return ni_oth.id == to; }
                            ) != dfly[ni.id].neighs.end();
                        }
                    );
                    
                    assert(it != dfly[ind].neighs.end());
                    it->out_qu.emplace(from, to);
                } else { ///(det) leaf -> host (== to).
                    auto it = std::find_if(dfly[ind].neighs.begin(), dfly[ind].neighs.end(), [&to](const NeighInfo& ni) { return ni.id == to; } );
                    assert(it != dfly[ind].neighs.end());
                    it->out_qu.emplace(from, to);
                }
            } else { ///sunt in spine-ul intermediar.
                assert(!is_node_host(ind));

                if (!is_switch_leaf(ind)) { ///spine.
                    auto it = std::find_if(
                        dfly[ind].neighs.begin(), dfly[ind].neighs.end(),
                        [&group_to](const NeighInfo& ni) { return ni.id / GROUP_SIZE == group_to; }
                    );

                    if (it != dfly[ind].neighs.end()) { ///(det) nu trebuie sa schimb spine-ul, am legatura directa catre group_to.
                        it->out_qu.emplace(from, to);
                    } else { ///(rand) spine -> leaf.
                        need_random.emplace_back(Packet(from, to), dfly[ind].neighs.begin(), dfly[ind].neighs.begin() + HALF_K);
                    }
                } else { ///(det) leaf -> spine.
                    auto it = std::find_if(
                        dfly[ind].neighs.begin(), dfly[ind].neighs.end(),
                        [&group_to, &dfly](const NeighInfo& ni) {
                            return std::find_if(
                                dfly[ni.id].neighs.begin(), dfly[ni.id].neighs.end(),
                                [&group_to](const NeighInfo& ni_oth) { return ni_oth.id / GROUP_SIZE == group_to; }
                            ) != dfly[ni.id].neighs.end();
                        }
                    );

                    assert(it != dfly[ind].neighs.end());
                    it->out_qu.emplace(from, to);
                }
            }
        }

        bb->make_picks(need_random);
    }
}

void end_step(std::array<Node, DFLY_SIZE>& dfly) {
    for (int i = 0; i < DFLY_SIZE; i++) {
        for (NeighInfo& ni: dfly[i].neighs) {
            ni.end_step_qu_sizes.push_back(ni.out_qu.size());
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: 1) <global topology configuration (random, configs/k_4/config_k_4_score_0.txt)>\n";
        std::cerr << "       2) <traffic pattern> (group_incast, host_incast, all_to_all)\n";
        std::cerr << "       3) <balls bins strategy> (greedy1, greedy2seq, greedy2par)\n";
        std::cerr << "       4) <cnt steps> (any integer >= 1)\n";
        std::cerr << "ex ./sim1 random group_incast greedy1 100\n";
        return 0;
    }

    std::string config(argv[1]), traffic_pattern(argv[2]), balls_bins(argv[3]);
    int cnt_steps = atoi(argv[4]);

    std::array<Node, DFLY_SIZE> dfly = {};

    generate_dfly(dfly, config);
    // dbg_dfly(dfly);

    // auto spine_cfg = dfly_get_spine_cfg(dfly);
    // dbgs2d(spine_cfg);

    // int score = dfly_state_score(spine_cfg);
    // dbgln(score);

    std::unique_ptr<TrafficPattern> tp;
    if (traffic_pattern == "group_incast") tp = std::make_unique<GroupIncast>(dfly, 0);
    else if (traffic_pattern == "host_incast") tp = std::make_unique<HostIncast>(dfly, 0);
    else if (traffic_pattern == "all_to_all") tp = std::make_unique<AllToAll>(dfly);
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

    for (int i = 0; i < DFLY_SIZE; i++) {
        std::cout << i << ' ' << dfly[i].neighs.size() << '\n';
        for (const NeighInfo& ni: dfly[i].neighs) {
            // std::cout << '\t';
            // for (int x: ni.end_step_qu_sizes) std::cout << x << ' ';
            // std::cout << '\n';
            std::cout << ni.end_step_qu_sizes.back() << ' ';
        }
        std::cout << '\n';
    }

    // std::cout << "cnt_delivered_packets = " << cnt_delivered_packets << '\n';
    // for (int i = 0; i < DFLY_SIZE; i++) {
    //     if (i == 0 || i / GROUP_SIZE != (i-1) / GROUP_SIZE) std::cerr << "---\ngroup " << i / GROUP_SIZE << '\n';

    //     std::cerr << "node " << i << ": " << (is_node_host(i)? "host": "switch");
    //     if (!is_node_host(i)) std::cerr << ", " << (is_switch_leaf(i)? "leaf": "spine");
    //     std::cerr << ". Final queue sizes: ";
    //     int sum_all = 0;
    //     for (const NeighInfo& ni: dfly[i].neighs) {
    //         std::cerr << ni.end_step_qu_sizes.back() << ", ";
    //         sum_all += ni.end_step_qu_sizes.back();
    //     }

    //     std::cerr << "sum = " << sum_all << '\n';
    // }

    return 0;
}
