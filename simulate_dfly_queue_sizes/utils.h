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

constexpr bool DEBUG = false;

constexpr int K = 8, HALF_K = K / 2, CNT_GROUPS = 1 + HALF_K * HALF_K, GROUP_SIZE = HALF_K * HALF_K + 2 * HALF_K, DFLY_SIZE = CNT_GROUPS * GROUP_SIZE;

struct Packet {
    int from, to;
    ///TODO: pentru REPS ar trebui culoare?

    Packet(int from, int to): from(from), to(to) {}
};

struct NeighInfo {
    int id;
    std::queue<Packet> out_qu;
    std::vector<int> end_step_qu_sizes;

    NeighInfo(int id): id(id) {}
    NeighInfo(): id(-1) {}
};

///host sau switch.
struct Node {
    std::vector<NeighInfo> neighs;

    Node() {
        neighs.reserve(K);
    }
};

void dbg_dfly(std::array<Node, DFLY_SIZE>& dfly);

bool is_node_host(int id);

bool is_switch_leaf(int id);

void generate_dfly(std::array<Node, DFLY_SIZE>& dfly, std::string type);

std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> dfly_get_spine_cfg(std::array<Node, DFLY_SIZE>& dfly);

int dfly_state_score(std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> spine_cfg);
