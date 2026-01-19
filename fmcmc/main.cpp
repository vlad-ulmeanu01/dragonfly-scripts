#include "utils.h"
#include "fmcmc.h"

int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "Usage: 1) <pheromone initial value>\n";
        std::cerr << "       2) <pheromone decay value rho>\n";
        std::cerr << "       3) <pheromone increase Q>\n";
        std::cerr << "       4) <pheromone alpha>\n";
        std::cerr << "       5) <use pheromone bounds?>\n";
        std::cerr << "       6) <batch size>\n";
        std::cerr << "       7) <cnt epochs>\n";
        std::cerr << "ex ./main 1.0 0.1 10.0 1.0 true 100 20\n";
        return 0;
    }

    double PHEROM_INIT = atof(argv[1]), PHEROM_DECAY = atof(argv[2]), PHEROM_Q = atof(argv[3]), PHEROM_EXP = atof(argv[4]);
    bool USE_PHEROM_BOUNDS = std::string(argv[5]) == "true";
    int BATCH_SIZE = atoi(argv[6]), CNT_EPOCHS = atoi(argv[7]);

    Logger logger(
        "/home/vlad/Desktop/Probleme/LaburiSOS/Proiect/logs/" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) +
        ".json"
    );

    STORE(logger, K);
    STORE(logger, HALF_K);
    STORE(logger, CNT_GROUPS);
    STORE(logger, GROUP_SIZE);
    STORE(logger, FMCMC_SIZE);
    STORE(logger, MAX_FLOW);
    STORE(logger, MIN_COST);

    STORE(logger, PHEROM_INIT);
    STORE(logger, PHEROM_DECAY);
    STORE(logger, PHEROM_Q);
    STORE(logger, PHEROM_EXP);
    STORE(logger, USE_PHEROM_BOUNDS);
    STORE(logger, BATCH_SIZE);
    STORE(logger, CNT_EPOCHS);

    Fmcmc fmcmc(FMCMC_SIZE, 0, FMCMC_SIZE-1, PHEROM_EXP);

    ///start -> noduri grupe.
    for (int i = 1; i <= CNT_GROUPS; i++) {
        fmcmc.add_edge(0, i, 3 * HALF_K * HALF_K * (HALF_K - 1) / 2, false, PHEROM_INIT);
    }
    
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
                        fmcmc.add_edge(1 + group_id, j + off_j, 3, true, PHEROM_INIT);

                        fmcmc.add_edge(j + off_j, z + x - (x > group_id), 1, false, PHEROM_INIT);
                        fmcmc.add_edge(j + off_j, z + y - (y > group_id), 1, false, PHEROM_INIT);

                        fmcmc.add_edge(j + off_j, t + off_t1 + y-x-1, 1, false, PHEROM_INIT);

                        off_j++;
                    }
                }
            }
        }

        for (int off_z = 0; off_z < CNT_GROUPS; off_z++) {
            if (off_z != group_id) {
                fmcmc.add_edge(z + off_z - (off_z > group_id), fmcmc.node_end, HALF_K - 1, false, PHEROM_INIT);
            }
        }

        j += (CNT_GROUPS - 1) * (CNT_GROUPS - 2) / 2;
        z += CNT_GROUPS - 1;
    }
    
    for (int t = fmcmc.node_end - CNT_GROUPS * (CNT_GROUPS - 1) / 2; t < fmcmc.node_end; t++) {
        fmcmc.add_edge(t, fmcmc.node_end, HALF_K - 1, false, PHEROM_INIT);
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

        if (USE_PHEROM_BOUNDS) {
            for (int i = 0; i < fmcmc.cnt_nodes; i++) {
                for (Edge& e: fmcmc.neighs[i]) e.pherom = std::max(pherom_lo, std::min(pherom_hi, e.pherom));
            }
        }

        STORE(logger, mean_deg);
        STORE(logger, pherom_lo);
        STORE(logger, pherom_hi);
        STORE(logger, best_cost);
        STORE(logger, bo_cost);

        // std::cerr << "Epoch " << epoch_id << " ended.\n";
        // std::cerr << "best cost = " << best_cost << ", best overall cost = " << bo_cost << ", target cost = " << MIN_COST << '\n';
        // std::cerr << "---\n";
    }

    int bo_flow = fmcmc.get_flow(bo_used_edges);

    STORE(logger, bo_flow);
    // std::cerr << "bo_flow = " << bo_flow << ", vs max flow = " << MAX_FLOW << '\n';

    for (const Edge& e: bo_used_edges) {
        logger.store("bo_from", e.from);
        logger.store("bo_to", e.to);
        logger.store("bo_flow", e.flow);
        logger.store("bo_cap", e.cap);
        logger.store("bo_has_con", e.has_con);
        
        // dbg(e.from) dbg(e.to) dbg(e.flow) dbg(e.cap) dbg(e.has_con) dbgln(e.pherom)
    }

    for (int i = 0; i < fmcmc.cnt_nodes; i++) {
        for (const Edge& e: fmcmc.neighs[i]) {
            logger.store("pherom_from", e.from);
            logger.store("pherom_to", e.to);
            logger.store("pherom", e.pherom);
        }
    }

    return 0;
}
