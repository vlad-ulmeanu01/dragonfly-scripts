#include "utils.h"


Stats::Stats(double mean, double std): mean(mean), std(std) {}

Stats::Stats(const std::vector<int>& v) {
    mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();

    std = 0.0;
    for (int i = 0; i < (int)v.size(); i++) std += (v[i] - mean) * (v[i] - mean);
    std = sqrt(std / ((int)v.size() - 1));
}

std::ostream& operator << (std::ostream& out, const Stats& s) {
    out << std::fixed << std::setprecision(3) << s.mean << ' ' << s.std;
    return out;
}


DflyPlusMaxHosts::DflyPlusMaxHosts(int K, int PACKS_GEN_PER_STEP, int WIRE_TRANS_PER_STEP, std::string cfg_type):
    K(K), PACKS_GEN_PER_STEP(PACKS_GEN_PER_STEP), WIRE_TRANS_PER_STEP(WIRE_TRANS_PER_STEP), cfg_type(cfg_type),
    HALF_K(K / 2), CNT_GROUPS(1 + HALF_K * HALF_K), GROUP_SIZE(HALF_K * HALF_K + 2 * HALF_K), DFLY_SIZE(CNT_GROUPS * GROUP_SIZE),
    topo(DFLY_SIZE), spine_cfg(CNT_GROUPS, std::vector<int>(CNT_GROUPS - 1))
{
    for (int i = 0; i < DFLY_SIZE; i++) topo[i].reserve(K);

    for (int group_id = 0, offset = 0; group_id < CNT_GROUPS; group_id++, offset += GROUP_SIZE) {
        ///[offset, offset + HALF_K**2) sunt terminale.
        for (int i = offset, leaf = offset + HALF_K * HALF_K, j = 0; i < offset + HALF_K * HALF_K; i++) {
            topo[i].emplace_back(leaf);
            topo[leaf].emplace_back(i);
            j++;
            if (j >= HALF_K) {
                j = 0;
                leaf++;
            }
        }

        ///ultima jumatate de conexiuni leaves si prima jumatate de conn spine.
        for (int leaf = offset + HALF_K * HALF_K; leaf < offset + HALF_K * HALF_K + HALF_K; leaf++) {
            for (int j = 0; j < HALF_K; j++) {
                topo[leaf].emplace_back(offset + HALF_K * HALF_K + HALF_K + j);
                topo[offset + HALF_K * HALF_K + HALF_K + j].emplace_back(leaf);
            }
        }
    }

    std::random_device rd;
    std::mt19937 mt(DEBUG? 0: rd());
    std::ifstream cfg_in;
    if (cfg_type != "random") cfg_in = std::ifstream(cfg_type);

    for (int group_id = 0; group_id < CNT_GROUPS; group_id++) {
        if (cfg_type == "random") {
            std::iota(spine_cfg[group_id].begin(), spine_cfg[group_id].begin() + group_id, 0);
            std::iota(spine_cfg[group_id].begin() + group_id, spine_cfg[group_id].end(), group_id + 1);
            std::shuffle(spine_cfg[group_id].begin(), spine_cfg[group_id].end(), mt);
        } else {
            for (int i = 0; i < CNT_GROUPS - 1; i++) cfg_in >> spine_cfg[group_id][i];
        }
    }

    for (int group_id = 0, offset = 0; group_id < CNT_GROUPS; group_id++, offset += GROUP_SIZE) {
        ///ultima jumatate de conn spine, e.g. cu alte grupuri.
        for (int spine = offset + HALF_K * HALF_K + HALF_K, j = 0; spine < offset + HALF_K * HALF_K + 2 * HALF_K; spine++) {
            for (int i = 0; i < HALF_K; i++) {
                int oth_group_id = spine_cfg[group_id][j++];
                topo[spine].emplace_back(
                    oth_group_id * GROUP_SIZE + HALF_K * HALF_K + HALF_K +
                    (std::find(spine_cfg[oth_group_id].begin(), spine_cfg[oth_group_id].end(), group_id) - spine_cfg[oth_group_id].begin()) / HALF_K
                );
            }
        }
    }
}


bool DflyPlusMaxHosts::is_node_host(int id) {
    int group_id = id / GROUP_SIZE;
    return id < group_id * GROUP_SIZE + HALF_K * HALF_K;
}


///stim ca id apartine unui switch.
bool DflyPlusMaxHosts::is_switch_leaf(int id) {
    int group_id = id / GROUP_SIZE;
    return id < group_id * GROUP_SIZE + HALF_K * HALF_K + HALF_K;
}


int DflyPlusMaxHosts::get_score() {
    std::vector<std::vector<int>> freq(CNT_GROUPS, std::vector<int>(CNT_GROUPS));

    for (int i = 0; i < CNT_GROUPS; i++) {
        for (int batch_offset = 0; batch_offset < CNT_GROUPS - 1; batch_offset += HALF_K) {
            for (int j = batch_offset; j < batch_offset + HALF_K; j++) {
                for (int z = j+1; z < batch_offset + HALF_K; z++) {
                    int x = spine_cfg[i][j], y = spine_cfg[i][z];
                    freq[std::min(x, y)][std::max(x, y)]++;
                }
            }
        }
    }

    // dbgs2d(freq);

    int scor = 0;
    for (int i = 0; i < CNT_GROUPS; i++) {
        for (int j = i+1; j < CNT_GROUPS; j++) scor -= abs(freq[i][j] - (HALF_K - 1));
    }

    return scor;
}


void DflyPlusMaxHosts::dbg_topo() {
    for (int i = 0; i < DFLY_SIZE; i++) {
        if (i == 0 || i / GROUP_SIZE != (i-1) / GROUP_SIZE) std::cerr << "---\ngroup " << i / GROUP_SIZE << '\n';
        std::cerr << "node " << i << ": " << (is_node_host(i)? "host": "switch") << ", " << topo[i].size() << " neighs";
        if (!is_node_host(i)) std::cerr << ", " << (is_switch_leaf(i)? "leaf": "spine");
        std::cerr << '\n';
        for (NeighInfo& ni: topo[i]) {
            int group = ni.id / GROUP_SIZE;
            std::cerr << '\t' << ni.id << ", from group " << group << ", " << (is_node_host(i)? "host": "switch");
            if (!is_node_host(ni.id)) std::cerr << ", " << (is_switch_leaf(ni.id)? "leaf": "spine");
            std::cerr << '\n';
        }
    }
}


/// reconstruieste spine_cfg din topo.
std::vector<std::vector<int>> DflyPlusMaxHosts::dbg_get_spine_cfg() {
    std::vector<std::vector<int>> spine_cfg_r(CNT_GROUPS, std::vector<int>(CNT_GROUPS - 1));
    std::vector<int> size(CNT_GROUPS);

    for (int group_offset = 0, group_id = 0; group_offset < DFLY_SIZE; group_offset += GROUP_SIZE, group_id++) {
        for (int spine = group_offset + HALF_K * HALF_K + HALF_K; spine < group_offset + HALF_K * HALF_K + 2 * HALF_K; spine++) {
            for (NeighInfo& ni: topo[spine]) {
                if (!is_switch_leaf(ni.id)) {
                    spine_cfg_r[group_id][size[group_id]] = ni.id / GROUP_SIZE;
                    size[group_id]++;
                }
            }
        }
    }

    return spine_cfg_r;
}
