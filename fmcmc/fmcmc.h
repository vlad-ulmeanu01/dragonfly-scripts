#pragma once
#include "utils.h"

struct Edge {
    int from, to, flow, cap; ///from e implicit.
    bool has_con; ///daca are cost concav pe muchie. altfel e cost 0 pe muchie.
    double pherom;

    Edge (int from, int to, int cap, bool has_con, int pherom_init): from(from), to(to), flow(0), cap(cap), has_con(has_con), pherom(pherom_init) {}
};

struct Fmcmc { ///flux maxim de cost minim cu costuri concave pe muchii (si capacitati pe muchii).
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;
    int cnt_nodes, node_start, node_end;
    double pherom_exp;
    std::vector<std::vector<Edge>> neighs;
    std::vector<int> info_cnt_choices;

    Fmcmc (int cnt_nodes, int node_start, int node_end, double pherom_exp):
        mt(DEBUG? 0: rd()), dist(0, 1), cnt_nodes(cnt_nodes), node_start(node_start), node_end(node_end), pherom_exp(pherom_exp)
    {
        neighs.resize(cnt_nodes);
    }

    void add_edge(int from, int to, int cap, bool has_con, int pherom_init);

    int choice(int nod, std::vector<bool>& viz);

    void dfs(std::vector<int>& path, std::vector<bool>& viz, bool& find_amel, int& exces);

    std::pair<double, std::vector<Edge>> one_maxflow();

    int get_flow(std::vector<Edge>& used_edges);
};
