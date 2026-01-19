#include "fmcmc.h"
#include "utils.h"

void Fmcmc::add_edge(int from, int to, int cap, bool has_con, int pherom_init) {
    neighs[from].emplace_back(from, to, cap, has_con, pherom_init);
    neighs[to].emplace_back(to, from, 0, false, pherom_init);
}

///alege urmatorul nod intr-o plimbare a unui agent.
///nu poate alege un nod deja vizitat in cautare. nu poate alege un nod a carui capacitate nu poate fi crescuta.
int Fmcmc::choice(int nod, std::vector<bool>& viz) {
    std::vector<std::pair<double, int>> cand_scores_ids;

    int i = 0;
    for (const Edge& e: neighs[nod]) {
        if (!viz[e.to] && e.flow < e.cap) {
            double scor;
            if (pherom_exp == 1) scor = e.pherom;
            else scor = pow(e.pherom, pherom_exp);

            cand_scores_ids.emplace_back(scor, i);
        }
        i++;
    }

    info_cnt_choices.push_back(cand_scores_ids.size());
    if (cand_scores_ids.empty()) return -1;

    double sum_scor = 0, pref_sum = 0;
    for (auto& p: cand_scores_ids) sum_scor += p.first;

    for (auto& p: cand_scores_ids) {
        p.first /= sum_scor;
        pref_sum += p.first;
        p.first = pref_sum;
    }

    double r = dist(mt);
    int z = (int)cand_scores_ids.size() - 1;
    for (int pas = get_msb(cand_scores_ids.size()); pas; pas >>= 1) {
        if (z - pas >= 0 && cand_scores_ids[z-pas].first > r) z -= pas;
    }

    return cand_scores_ids[z].second;
}

void Fmcmc::dfs(std::vector<int>& path, std::vector<bool>& viz, bool& find_amel, int& exces) {
    viz[path.back()] = true;
    if (path.back() == node_end) {
        find_amel = true;
        return;
    }

    int i = choice(path.back(), viz);
    while (i != -1) {
        Edge& e = neighs[path.back()][i];
        path.push_back(e.to);
        
        int tmp_exces = std::min(exces, e.cap - e.flow);
        dfs(path, viz, find_amel, tmp_exces);
        
        if (find_amel) {
            exces = tmp_exces;
            return;
        }

        path.pop_back();
        i = choice(path.back(), viz);
    }
}

std::pair<double, std::vector<Edge>> Fmcmc::one_maxflow() {
    std::vector<bool> viz(cnt_nodes);
    std::vector<int> path;

    bool find_amel = true;
    while (find_amel) {
        std::fill(viz.begin(), viz.end(), false);
        find_amel = false;
        path = {node_start};

        int exces = inf;
        dfs(path, viz, find_amel, exces);

        if (find_amel) {
            for (int i = 0; i+1 < (int)path.size(); i++) {
                std::find_if(neighs[path[i]].begin(), neighs[path[i]].end(), [&path, &i](const Edge& e) { return e.to == path[i+1]; })->flow += exces;
                std::find_if(neighs[path[i+1]].begin(), neighs[path[i+1]].end(), [&path, &i](const Edge& e) { return e.to == path[i]; })->flow -= exces;
            }
        }
    }

    double cost = 0;
    std::vector<Edge> flow_components;
    for (int i = 0; i < cnt_nodes; i++) {
        for (Edge& e: neighs[i]) {
            if (e.flow > 0) {
                if (e.has_con) cost += con_costs[e.flow];
                flow_components.emplace_back(e);
            }
            e.flow = 0;
        }
    }

    return std::make_pair(cost, flow_components);
}

int Fmcmc::get_flow(std::vector<Edge>& used_edges) {
    int flow = 0;
    for (Edge& e: used_edges) {
        if (e.from == node_start) flow += e.flow;
    }
    return flow;
}
