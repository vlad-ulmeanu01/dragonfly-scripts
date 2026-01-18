#include "utils.h"
#include "fmcmc.h"

int main() {
    Fmcmc fmcmc(FMCMC_SIZE, 0, FMCMC_SIZE-1);

    // dbg(FMCMC_SIZE) dbg(MAX_FLOW) dbgln(MIN_COST)

    ///start -> noduri grupe.
    for (int i = 1; i <= CNT_GROUPS; i++) {
        fmcmc.add_edge(0, i, 3 * HALF_K * HALF_K * (HALF_K - 1) / 2, false);
    }
    
    // dbg("j") dbgln(CNT_GROUPS + 1)
    // dbg("z") dbgln(CNT_GROUPS + 1 + CNT_GROUPS * (CNT_GROUPS - 1) * (CNT_GROUPS - 2) / 2)
    // dbg("t") dbgln(fmcmc.node_end - CNT_GROUPS * (CNT_GROUPS - 1) / 2)

    ///pentru fiecare grupa:
    ///j = unde incep (per grup) (1, 2), (1, 3), (1, 5), ..., (9, 10)
    ///z = unde incep (per grup) 1, 2, 3, 5, .., 10
    ///t = unde incep (1, 2), (1, 3), (1, 4), (1, 5), .., (9, 10).
    for (int group_id = 0,
         j = CNT_GROUPS + 1,
         z = CNT_GROUPS + 1 + CNT_GROUPS * (CNT_GROUPS - 1) * (CNT_GROUPS - 2) / 2,
         t = fmcmc.node_end - CNT_GROUPS * (CNT_GROUPS - 1) / 2;
         group_id < CNT_GROUPS; group_id++) {

        for (int x = 0, off_j = 0, off_t1 = 0; x < CNT_GROUPS; off_t1 += (CNT_GROUPS - 1 - x), x++) {
            if (x != group_id) {
                for (int y = x + 1; y < CNT_GROUPS; y++) {
                    if (y != group_id) {
                        fmcmc.add_edge(1 + group_id, j + off_j, 3, true);

                        fmcmc.add_edge(j + off_j, z + x - (x > group_id), 1, false);
                        fmcmc.add_edge(j + off_j, z + y - (y > group_id), 1, false);

                        fmcmc.add_edge(j + off_j, t + off_t1 + y-x-1, 1, false);

                        off_j++;
                    }
                }
            }
        }

        for (int off_z = 0; off_z < CNT_GROUPS; off_z++) {
            if (off_z != group_id) {
                fmcmc.add_edge(z + off_z - (off_z > group_id), fmcmc.node_end, HALF_K - 1, false);
            }
        }

        j += (CNT_GROUPS - 1) * (CNT_GROUPS - 2) / 2;
        z += CNT_GROUPS - 1;
    }
    
    for (int t = fmcmc.node_end - CNT_GROUPS * (CNT_GROUPS - 1) / 2; t < fmcmc.node_end; t++) {
        fmcmc.add_edge(t, fmcmc.node_end, HALF_K - 1, false);
    }

    double bo_cost = inf; /// best_overall.
    std::vector<Edge> bo_used_edges;

    std::cout << std::fixed << std::setprecision(3);
    for (int epoch_id = 1; epoch_id <= CNT_EPOCHS; epoch_id++) {
        double best_cost = inf;
        std::vector<Edge> best_used_edges;

        for (int _ = 0; _ < BATCH_SIZE; _++) {
            auto [cost, used_edges] = fmcmc.one_maxflow();
            if (cost < best_cost) {
                best_cost = cost;
                best_used_edges = used_edges;
            }
        }

        if (best_cost < bo_cost) {
            bo_cost = best_cost;
            bo_used_edges = best_used_edges;
        }

        double mean_deg = std::accumulate(fmcmc.info_cnt_choices.begin(), fmcmc.info_cnt_choices.end(), 0.0) / fmcmc.info_cnt_choices.size();
        fmcmc.info_cnt_choices.clear();

        double pherom_hi = 1 / (PHEROM_DECAY * best_cost), pherom_lo = pherom_hi * (1 - P_BEST_1N) / ((mean_deg - 1) * P_BEST_1N);

        for (int i = 0; i < fmcmc.cnt_nodes; i++) {
            for (Edge& e: fmcmc.neighs[i]) e.pherom *= (1 - PHEROM_DECAY);
        }

        double q_div = PHEROM_Q / best_cost;
        for (Edge& e: best_used_edges) {
            std::find_if(fmcmc.neighs[e.from].begin(), fmcmc.neighs[e.from].end(), [&e](const Edge& f) { return f.to == e.to; })->pherom += q_div;
        }

        for (int i = 0; i < fmcmc.cnt_nodes; i++) {
            for (Edge& e: fmcmc.neighs[i]) e.pherom = std::max(pherom_lo, std::min(pherom_hi, e.pherom));
        }

        std::cout << "Epoch " << epoch_id << " ended.\n";
        std::cout << "best cost = " << best_cost << ", best overall cost = " << bo_cost << ", target cost = " << MIN_COST << '\n';
        std::cout << "---\n";
    }

    int bo_flow = fmcmc.get_flow(bo_used_edges);
    std::cout << "bo_flow = " << bo_flow << ", vs max flow = " << MAX_FLOW << '\n';

    for (const Edge& e: bo_used_edges) {
        dbg(e.from) dbg(e.to) dbg(e.flow) dbg(e.cap) dbg(e.has_con) dbgln(e.pherom)
    }

    return 0;
}
