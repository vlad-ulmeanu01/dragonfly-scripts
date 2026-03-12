#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <utility>
#include <numeric>
#include <iomanip>
#include <random>
#include <vector>
#include <memory>
#include <queue>
#include <map>

#include <cassert>

#define aaa system("read -r -p \"Press enter to continue...\" key");
#define dbg(x) std::cerr << (#x) << ": " << (x) << ", ";
#define dbgln(x) std::cerr << (#x) << ": " << (x) << '\n';
#define dbga(x,n) { std::cerr << (#x) << "[]: "; for(int _ = 0; _ < n; _++) std::cerr << x[_] << ' '; std::cerr<<'\n'; }
#define dbgs(x) { std::cerr << (#x) << "[stl]: "; for(auto _: x) std::cerr << _ << ' '; std::cerr<<'\n'; }
#define dbgp(x) { std::cerr << (#x) << ": " << x.first << ' ' << x.second << '\n'; }
#define dbgs2d(x) { std::cerr << (#x) << "[stl 2d]:\n";  for(const auto& y: x) { for (auto _: y) std::cerr << _ << ' '; std::cerr << '\n'; } std::cerr << "---\n"; }
#define dbgsp(x) { std::cerr << (#x) << "[stl pair]:\n"; for(auto _: x) std::cerr << _.first << ' '<< _.second << '\n'; }

constexpr bool DEBUG = false;


struct Stats {
    double mean, std;

    Stats(double mean, double std);
    Stats(const std::vector<int>& v);
};

std::ostream& operator << (std::ostream& out, const Stats& s);


struct Packet {
    int from, to;
    ///TODO: pentru REPS ar trebui culoare?

    Packet(int from, int to);
};


struct PacketQueue {
    int GROUP_SIZE;
    std::queue<Packet> qu;
    int group_now; ///coada apartine de un sw. in ce grup e switch-ul?
    int cnt_oog_packs; ///cate pachete out-of-group detin (i.e. pentru care group_now nu e nici from nici to).

    PacketQueue(int GROUP_SIZE, int group_now);
    PacketQueue();

    void push(const Packet& p);
    void pop();
    const Packet& front();
    size_t size();
    bool empty();

    template <typename... Args>
    void emplace(Args&&... args) {
        push(Packet(std::forward<Args>(args)...));
    }
};


struct NeighInfo {
    int id;
    PacketQueue out_qu;
    std::vector<int> end_step_out_qu_sizes;

    NeighInfo(int GROUP_SIZE, int id);
    NeighInfo();
};


struct DflyPlusMaxHosts {
    int K; ///switch radix.
    int PACKS_GEN_PER_STEP, WIRE_TRANS_PER_STEP;
    
    std::string cfg_type; ///cfg_type in ["random", (config file name)]

    /// daca un host poate produce 1 pachet/step, atunci pentru un incast pot fi generate (G-1)(K/2)**2 = (K/2)**4 pachete ce trebuie receptionate prin (K/2)**2 host-uri.
    /// deci pentru o impartire perfecta a traficului WIRE_TRANS_PER_STEP ar trebui sa fie de ajuns.

    int HALF_K, CNT_GROUPS, GROUP_SIZE, DFLY_SIZE;

    ///host sau switch.
    std::vector<std::vector<NeighInfo>> topo; ///topo[DFLY_SIZE][K]: NeighInfo.
    std::vector<std::vector<int>> spine_cfg;

    DflyPlusMaxHosts(int K, int PACKS_GEN_PER_STEP, int WIRE_TRANS_PER_STEP, std::string cfg_type);

    bool is_node_host(int id);
    bool is_switch_leaf(int id);
    int get_score();

    void dbg_topo();
    std::vector<std::vector<int>> dbg_get_spine_cfg();
};
