#include "utils.h"

void dbg_dfly(std::array<Node, DFLY_SIZE>& dfly) {
    for (int i = 0; i < DFLY_SIZE; i++) {
        if (i == 0 || i / GROUP_SIZE != (i-1) / GROUP_SIZE) std::cerr << "---\ngroup " << i / GROUP_SIZE << '\n';
        std::cerr << "node " << i << ": " << (is_node_host(i)? "host": "switch") << ", " << dfly[i].neighs.size() << " neighs";
        if (!is_node_host(i)) std::cerr << ", " << (is_switch_leaf(i)? "leaf": "spine");
        std::cerr << '\n';
        for (NeighInfo& ni: dfly[i].neighs) {
            int group = ni.id / GROUP_SIZE;
            std::cerr << '\t' << ni.id << ", from group " << group << ", " << (is_node_host(i)? "host": "switch");
            if (!is_node_host(ni.id)) std::cerr << ", " << (is_switch_leaf(ni.id)? "leaf": "spine");
            std::cerr << '\n';
        }
    }
}

bool is_node_host(int id) {
    int group_id = id / GROUP_SIZE;
    return id < group_id * GROUP_SIZE + HALF_K * HALF_K;
}

///stim ca id apartine unui switch.
bool is_switch_leaf(int id) {
    int group_id = id / GROUP_SIZE;
    return id < group_id * GROUP_SIZE + HALF_K * HALF_K + HALF_K;
}

void generate_dfly(std::array<Node, DFLY_SIZE>& dfly, std::string type) {
    ///type in ["random", (config file name)]

    for (int group_id = 0, offset = 0; group_id < CNT_GROUPS; group_id++, offset += GROUP_SIZE) {
        ///[offset, offset + HALF_K**2) sunt terminale.
        for (int i = offset, leaf = offset + HALF_K * HALF_K, j = 0; i < offset + HALF_K * HALF_K; i++) {
            dfly[i].neighs.emplace_back(leaf);
            dfly[leaf].neighs.emplace_back(i);
            j++;
            if (j >= HALF_K) {
                j = 0;
                leaf++;
            }
        }

        ///ultima jumatate de conexiuni leaves si prima jumatate de conn spine.
        for (int leaf = offset + HALF_K * HALF_K; leaf < offset + HALF_K * HALF_K + HALF_K; leaf++) {
            for (int j = 0; j < HALF_K; j++) {
                dfly[leaf].neighs.emplace_back(offset + HALF_K * HALF_K + HALF_K + j);
                dfly[offset + HALF_K * HALF_K + HALF_K + j].neighs.emplace_back(leaf);
            }
        }
    }

    std::random_device rd;
    std::mt19937 mt(DEBUG? 0: rd());
    std::ifstream cfg_in;
    if (type != "random") cfg_in = std::ifstream(type);

    std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> spine_cfg;
    for (int group_id = 0; group_id < CNT_GROUPS; group_id++) {
        if (type == "random") {
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
                dfly[spine].neighs.emplace_back(
                    oth_group_id * GROUP_SIZE + HALF_K * HALF_K + HALF_K +
                    (std::find(spine_cfg[oth_group_id].begin(), spine_cfg[oth_group_id].end(), group_id) - spine_cfg[oth_group_id].begin()) / HALF_K
                );
            }
        }
    }
}

std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> dfly_get_spine_cfg(std::array<Node, DFLY_SIZE>& dfly) {
    std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> spine_cfg;
    std::array<int, CNT_GROUPS> size = {};
    for (int group_offset = 0, group_id = 0; group_offset < DFLY_SIZE; group_offset += GROUP_SIZE, group_id++) {
        for (int spine = group_offset + HALF_K * HALF_K + HALF_K; spine < group_offset + HALF_K * HALF_K + 2 * HALF_K; spine++) {
            for (NeighInfo& ni: dfly[spine].neighs) {
                if (!is_switch_leaf(ni.id)) {
                    spine_cfg[group_id][size[group_id]] = ni.id / GROUP_SIZE;
                    size[group_id]++;
                }
            }
        }
    }

    return spine_cfg;
}

int dfly_state_score(std::array<std::array<int, CNT_GROUPS - 1>, CNT_GROUPS> spine_cfg) {
    std::array<std::array<int, CNT_GROUPS>, CNT_GROUPS> freq = {};
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
