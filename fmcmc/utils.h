#pragma once

#include <bits/stdc++.h>
#define aaa system("read -r -p \"Press enter to continue...\" key");
#define dbg(x) std::cerr << (#x) << ": " << (x) << ", ";
#define dbgln(x) std::cerr << (#x) << ": " << (x) << '\n';
#define dbga(x,n) { std::cerr << (#x) << "[]: "; for(int _ = 0; _ < n; _++) std::cerr << x[_] << ' '; std::cerr<<'\n'; }
#define dbgs(x) { std::cerr << (#x) << "[stl]: "; for(auto _: x) std::cerr << _ << ' '; std::cerr<<'\n'; }
#define dbgp(x) { std::cerr << (#x) << ": " << x.first << ' ' << x.second << '\n'; }
#define dbgs2d(x) { std::cerr << (#x) << "[stl 2d]:\n";  for(const auto& y: x) { for (auto _: y) std::cerr << _ << ' '; std::cerr << '\n'; } std::cerr << "---\n"; }
#define dbgsp(x) { std::cerr << (#x) << "[stl pair]:\n"; for(auto _: x) std::cerr << _.first << ' '<< _.second << '\n'; }

constexpr bool DEBUG = true;

constexpr double HEUR_EXP = 1.0;

const int inf = (1 << 30);

const std::array<double, 4> con_costs = {0, log(2), log(3), log(4)};

constexpr int K = 6, HALF_K = K / 2, CNT_GROUPS = 1 + HALF_K * HALF_K, GROUP_SIZE = HALF_K * HALF_K + 2 * HALF_K, DFLY_SIZE = CNT_GROUPS * GROUP_SIZE;
constexpr int FMCMC_SIZE = 1 + CNT_GROUPS * ((CNT_GROUPS-1) * CNT_GROUPS + 2) / 2 + CNT_GROUPS * (CNT_GROUPS - 1) / 2 + 1;
constexpr int MAX_FLOW = CNT_GROUPS * 3 * HALF_K * HALF_K * (HALF_K - 1) / 2;
constexpr double MIN_COST = CNT_GROUPS * HALF_K * HALF_K * (HALF_K - 1) / 2 * con_costs[3];

constexpr double CNT_EDGES_PATH_START_END = 4, P_BEST_1N = 0.47287; /// 0.05 ** (1 / 4).
// constexpr double PHEROM_INIT = 1.0, PHEROM_DECAY = 0.1, PHEROM_Q = 10.0;
// constexpr int BATCH_SIZE = 1000, CNT_EPOCHS = 100;

int get_msb(int x);

struct Logger {
    std::ofstream fout;
    std::map<std::string, std::vector<double>> ht;

    Logger(std::string name): fout(name) {}

    void store(std::string var_name, double x);

    ~Logger();
};
#define STORE(logger, x) logger.store(std::string(#x), x)
