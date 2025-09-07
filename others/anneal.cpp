#include <bits/stdc++.h>
#define aaa system("read -r -p \"Press enter to continue...\" key");
#define dbg(x) std::cerr<<(#x)<<": "<<(x)<<'\n',aaa
#define dbga(x,n) std::cerr<<(#x)<<"[]: ";for(int _=0;_<n;_++)std::cerr<<x[_]<<' ';std::cerr<<'\n',aaa
#define dbgs(x) std::cerr<<(#x)<<"[stl]: ";for(auto _:x)std::cerr<<_<<' ';std::cerr<<'\n',aaa
#define dbgp(x) std::cerr<<(#x)<<": "<<x.first<<' '<<x.second<<'\n',aaa
#define dbgsp(x) std::cerr<<(#x)<<"[stl pair]:\n";for(auto _:x)std::cerr<<_.first<<' '<<_.second<<'\n';aaa
#define TIMER(x) stop = std::chrono::steady_clock::now(); std::cerr << std::fixed << std::setprecision(3) << x << ": " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() * 1e-6 << "s\n"; start = std::chrono::steady_clock::now();

const int k = 8, g = k*k / 4 + 1;
const int expected = k/2-1, expected_min = expected - 0, expected_max = expected + 0;
const double temp_init = 1e4, temp_upd = 1 - 1e-4, temp_min = 0.5;
const int power = 1;
const int double_cnt_every_epochs = 10000;
int cnt_next_values = 100;

struct PotentialNextState {
    int z, i, j, new_cost;
    long double proba;

    PotentialNextState(int z, int i, int j, int new_cost, double proba): z(z), i(i), j(j), new_cost(new_cost), proba(proba) {}
};

int lgput(int a, int b) {
    int p2 = 1, ans = 1;
    while (b) {
        if (b & p2) ans *= a, b ^= p2;
        a *= a;
        p2 <<= 1;
    }
    return ans;
}

int get_cost(std::array<std::array<int, g>, g> &spine_used) {
    int cost = 0;
    for (int i = 0; i < g; i++) {
        for (int j = i+1; j < g; j++) {
            int curr_cost = 0;
            for (int z = 0; z < g; z++) {
                curr_cost += (z != i && z != j && spine_used[i][z] == spine_used[j][z]);
            }
            
            cost += (curr_cost < expected_min) * lgput(expected_min - curr_cost, power);
            cost += (curr_cost > expected_max) * lgput(curr_cost - expected_max, power);
        }
    }

    return cost;
}

int miniupd_cost(int z, int ij, int sgn, int cost,
    std::array<std::array<int, g>, g> &spine_used, std::array<std::array<std::unordered_set<int>, k/2>, g> &groups_in_spine,
    std::array<std::array<int, g>, g> &cnt_commons
) {
    for (int a: groups_in_spine[z][spine_used[ij][z]]) {
        int curr_cnt = cnt_commons[std::min(ij, a)][std::max(ij, a)];
        cost -= ((curr_cnt < expected_min) * lgput(expected_min - curr_cnt, power) + (curr_cnt > expected_max) * lgput(curr_cnt - expected_max, power));
        curr_cnt += sgn;
        cost += ((curr_cnt < expected_min) * lgput(expected_min - curr_cnt, power) + (curr_cnt > expected_max) * lgput(curr_cnt - expected_max, power));
        cnt_commons[std::min(ij, a)][std::max(ij, a)] = curr_cnt;
    }

    return cost;
}

/// returns the updated cost.
int do_swap(
    int z, int i, int j, int cost,
    std::array<std::array<int, g>, g> &spine_used, std::array<std::array<std::unordered_set<int>, k/2>, g> &groups_in_spine,
    std::array<std::array<int, g>, g> &cnt_commons
) {
    groups_in_spine[z][spine_used[i][z]].erase(i);
    groups_in_spine[z][spine_used[j][z]].erase(j);

    cost = miniupd_cost(z, i, -1, cost, spine_used, groups_in_spine, cnt_commons);
    cost = miniupd_cost(z, j, -1, cost, spine_used, groups_in_spine, cnt_commons);

    std::swap(spine_used[i][z], spine_used[j][z]);

    cost = miniupd_cost(z, i, 1, cost, spine_used, groups_in_spine, cnt_commons);
    cost = miniupd_cost(z, j, 1, cost, spine_used, groups_in_spine, cnt_commons);

    groups_in_spine[z][spine_used[i][z]].insert(i);
    groups_in_spine[z][spine_used[j][z]].insert(j);

    return cost;
}

int main() {
    auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();
    
    uint32_t seed = time(NULL);
    std::cout << "seed = " << seed << '\n';
    srand(seed);

    /// spine_used[i][j] = what spine router is used by group i in group j. default is -1 for [i][i].
    std::array<std::array<int, g>, g> spine_used = {};
    for (int i = 0; i < g; i++) for (int j = 0; j < g; j++) spine_used[i][j] = -1;

    /// groups_in_spine[i][j] = group i's j-th spine: {groups that use it}.
    std::array<std::array<std::unordered_set<int>, k/2>, g> groups_in_spine = {};
    
    /// cnt_commons[i][j] = in how many groups do group i and group j share the spine.
    std::array<std::array<int, g>, g> cnt_commons = {};

    // for (int i = 0; i < g; i++) {
    //     for (int j = 0; j < g; j++) {
    //         if (i != j) {
    //             /// what out spine does group i use in group j.
    //             int best_commons_here = INT_MAX, best_z = -1;

    //             for (int z = 0; z < k/2; z++) {
    //                 if ((int)groups_in_spine[j][z].size() < k/2) {
    //                     int commons_here = 0;
    //                     for (int a: groups_in_spine[j][z]) commons_here += cnt_commons[std::min(i, a)][std::max(i, a)];
    //                     if (commons_here < best_commons_here) {
    //                         best_commons_here = commons_here;
    //                         best_z = z;
    //                     }
    //                 }
    //             }

    //             for (int a: groups_in_spine[j][best_z]) cnt_commons[std::min(i, a)][std::max(i, a)]++;
    //             groups_in_spine[j][best_z].insert(i);
    //             spine_used[i][j] = best_z;
    //         }
    //     }
    // }

    for (int i = 0; i < g; i++) {
        for (int j = 0; j < g; j++) {
            if (i != j) {
                /// what out spine does group i use in group j.
                int cnt_choices = 0;
                for (int z = 0; z < k/2; z++) cnt_choices += ((int)groups_in_spine[j][z].size() < k/2);

                int ind = rand() % cnt_choices;
                for (int z = 0; z < k/2 && ind >= 0; z++) {
                    if ((int)groups_in_spine[j][z].size() < k/2) {
                        ind--;
                        if (ind < 0) {
                            groups_in_spine[j][z].insert(i);
                            spine_used[i][j] = z;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < g; i++) {
        for (int j = i+1; j < g; j++) {
            for (int z = 0; z < g; z++) {
                cnt_commons[i][j] += (z != i && z != j && spine_used[i][z] == spine_used[j][z]);
            }
        }
    }

    double temp = temp_init;
    int cost = get_cost(spine_used), best_cost = cost;

    std::cout << "epoch = 0, best_cost = " << best_cost << '\n' << std::flush;
    TIMER("spine used init")
    
    int epoch = 1, best_epoch = 0;
    while (best_cost > 0) {
        int z, i, j;
        int new_cost;

        ///choose two groups that belong to different spines in group z.
        std::vector<PotentialNextState> next_states;
        std::unordered_set<int> have_cost;
        long double total_proba = 0;
        for (int _ = 0; _ < cnt_next_values; _++) {
            z = rand() % g;
            do {
                i = rand() % g;
                j = rand() % g;
            } while (z == i || z == j || spine_used[i][z] == spine_used[j][z]);

            new_cost = do_swap(z, i, j, cost, spine_used, groups_in_spine, cnt_commons);

            if (!have_cost.count(new_cost)) {
                have_cost.insert(new_cost);
                next_states.emplace_back(z, i, j, new_cost, expl(std::max((double)(best_cost - new_cost) / temp, -100.0)));
                total_proba += next_states.back().proba;
            }

            cost = do_swap(z, i, j, new_cost, spine_used, groups_in_spine, cnt_commons);
        }

        for (PotentialNextState &next_state: next_states) {
            next_state.proba /= total_proba;
        }

        std::sort(next_states.begin(), next_states.end(), [](const PotentialNextState &a, const PotentialNextState &b) {
            return a.proba > b.proba;
        });
        
        double r = (double)rand() / RAND_MAX;

        int ind = 0;
        while (ind + 1 < (int)next_states.size() && next_states[ind].proba < r) {
            r -= next_states[ind].proba;
            ind++;
        }

        z = next_states[ind].z; i = next_states[ind].i; j = next_states[ind].j;
        cost = do_swap(z, i, j, cost, spine_used, groups_in_spine, cnt_commons);

        if (cost < best_cost) {
            best_cost = cost;
            best_epoch = epoch;
            std::cout << "epoch = " << epoch << ", best_cost = " << best_cost << '\n' << std::flush;
            TIMER("new best_cost")
        }

        temp = std::max(temp * temp_upd, temp_min);
        epoch += (best_cost > 0);
        if (epoch - best_epoch >= double_cnt_every_epochs) {
            cnt_next_values *= 2;
            std::cout << "cnt_next_values is now " << cnt_next_values << '\n';
            best_epoch = epoch;
        }
    }

    std::ofstream fout(
        "spine_configs_anneal/cpp_k_" + std::to_string(k) + "_min_" + std::to_string(expected - expected_min) +
        "_max_" + std::to_string(expected_max - expected) + "_seed_" + std::to_string(seed) +
        "_epoch_" + std::to_string(epoch) + ".txt"
    );

    fout << "[\n";
    for (int i = 0; i < g; i++) {
        fout << "\t[";
        for (int j = 0; j < k/2; j++) {
            fout << "[";
            int ind = 0;
            for (int a: groups_in_spine[i][j]) {
                fout << a+1 << (ind+1 < k/2? ", ": "");
                ind++;
            }
            fout << "]" << (j+1 < k/2? ", ": "");
        }
        fout << "]" << (i+1 < g? ",\n": "\n");
    }
    fout << "]\n";

    assert(get_cost(spine_used) == 0);

    return 0;
}
