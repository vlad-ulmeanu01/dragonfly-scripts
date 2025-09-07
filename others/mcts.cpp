#include <algorithm>
#include <iostream>
#include <ostream>
#include <utility>
#include <numeric>
#include <iomanip>
#include <vector>
#include <thread>
#include <random>
#include <tuple>
#include <array>
#include <mutex>
#include <map>

#include <climits>
#include <cmath>

#define NAME(x) (#x)
#define aaa system("read -r -p \"Press enter to continue...\" key");
#define dbg(x) std::cerr<<(#x)<<": "<<(x)<<'\n',aaa
#define dbga(x,n) std::cerr<<(#x)<<"[]: ";for(int _=0;_<n;_++)std::cerr<<x[_]<<' ';std::cerr<<'\n',aaa
#define dbgs(x) std::cerr<<(#x)<<"[stl]: ";for(auto _:x)std::cerr<<_<<' ';std::cerr<<'\n',aaa
#define dbgp(x) std::cerr<<(#x)<<": "<<x.first<<' '<<x.second<<'\n',aaa
#define dbgsp(x) std::cerr<<(#x)<<"[stl pair]:\n";for(auto _:x)std::cerr<<_.first<<' '<<_.second<<'\n';aaa

const int inf = (1 << 30) - 1;

const double ucb_ct = 1.0 / (6 * sqrt(2.0)); //sqrt(2.0);
const int mean_random_scores[] = {-6, -42, -163, -462, -1079, -2208, -4089, -7035, -11420, -17689, -26345, -37982, -53309, -73052}; ///pt k = 4, 6, 8, .. 30.
const int k = 6, g = k*k/4 + 1, expected = k/2 - 1;
const int max_num_children = g * k/2 * (k/2 - 1) / 2 * k/2 * k/2;
const int playout_length = 500;

typedef std::array<std::array<std::array<int, k/2>, k/2>, g> ARR_GKK;
typedef std::array<std::array<int, g>, g> ARR_GG;

void print_arr_gkk(ARR_GKK &arr) {
    std::cout << NAME(arr) << " = [\n";
    for (int z = 0; z < g; z++) {
        std::cout << "    [";
        for (int i = 0; i < k/2; i++) {
            std::cout << "[";
            for (int j = 0; j < k/2; j++) std::cout << arr[z][i][j] + 1 << (j+1 < k/2? ", ": "");
            std::cout << "]" << (i+1 < k/2? ", ": "");
        }
        std::cout << "]" << (z+1 < g? ",\n": "\n");
    }
    std::cout << "]\n";
}

void print_arr_gg(ARR_GG &arr) {
    std::cout << NAME(arr) << " = [\n";
    for (int i = 0; i < g; i++) {
        std::cout << "    [";
        for (int j = 0; j < g; j++) std::cout << arr[i][j] << (j+1 < g? ", ": "");
        std::cout << "]" << (i+1 < g? ",\n": "\n");
    }
    std::cout << "]\n";
}


struct Transform {
    int seq, group_a, id_a, group_b, id_b;

    Transform(): seq(0), group_a(0), id_a(0), group_b(0), id_b(0) {}
    Transform(int seq, int group_a, int id_a, int group_b, int id_b): seq(seq), group_a(group_a), id_a(id_a), group_b(group_b), id_b(id_b) {}

    bool operator < (const Transform &oth) const {
        if (seq != oth.seq) return seq < oth.seq;
        if (group_a != oth.group_a) return group_a < oth.group_a;
        if (id_a != oth.id_a) return id_a < oth.id_a;
        if (group_b != oth.group_b) return group_b < oth.group_b;
        return id_b < oth.id_b;
    }
};

struct State {
    std::map<Transform, State *> children_transforms;
    int score, ts_score, times_visited;

    State(): score(-inf), ts_score(0), times_visited(0) {}

    void add_ts_score(int new_ts_score, int amt) {
        ///pot sa adaug mai multe scoruri deodata.
        ts_score += new_ts_score;
        times_visited += amt;
    }

    double fitness(int times_visited_parent) {
        ///+1 ai fit pt random sa fie 0. fit pt best o sa fie 1. e greu sa scada sub -1 fit.
        return (((double)ts_score / times_visited) / -mean_random_scores[(k-4)/2] + 1.0) + ucb_ct * sqrt(log(times_visited_parent) / times_visited);
    }
};

///modifica in-place sequences, pair_count si intoarce noul scor.
int transform_sequences(ARR_GKK &seqs, ARR_GG &pair_count, Transform trans, int score) {
    auto miniupd = [&seqs, &pair_count, &score](int sign, int seq_no, int group_no, int a, int b) {
        int var1 = seqs[seq_no][group_no][a], var2 = seqs[seq_no][group_no][b];
        if (var1 > var2) std::swap(var1, var2);

        score += abs(pair_count[var1][var2] - expected);
        pair_count[var1][var2] += sign;
        score -= abs(pair_count[var1][var2] - expected);
    };

    for (int sign = -1; sign <= 1; sign += 2) {
        for (int x = 0; x < k/2; x++) {
            if (x != trans.id_a) miniupd(sign, trans.seq, trans.group_a, trans.id_a, x);
            if (x != trans.id_b) miniupd(sign, trans.seq, trans.group_b, trans.id_b, x);
        }

        if (sign == -1) std::swap(seqs[trans.seq][trans.group_a][trans.id_a], seqs[trans.seq][trans.group_b][trans.id_b]);
    }

    return score;
}

void playout_worker(
    int mt_seed, int &max_score, std::mutex &max_mut,
    int root_score, State *playout_start, Transform trans,
    ARR_GKK tmp_seqs, ARR_GG tmp_pair_count
) {
    std::mt19937 mt(mt_seed);
    std::uniform_int_distribution<int> dist_g(0, g-1);
    std::uniform_int_distribution<int> dist_k(0, k/2-1);

    ARR_GKK local_best_seqs = tmp_seqs;

    playout_start->score = transform_sequences(tmp_seqs, tmp_pair_count, trans, root_score); ///trans e actiunea root -> playout_start.

    int tmp_score = playout_start->score, tmp_max_score = tmp_score;
    for (int _ = 0; _ < playout_length; _++) {
        int z = dist_g(mt), i = dist_k(mt), a = dist_k(mt), j = dist_k(mt), b = dist_k(mt);
        while (j == i) j = dist_k(mt);
        if (j < i) std::swap(i, j);

        tmp_score = transform_sequences(tmp_seqs, tmp_pair_count, Transform(z, i, a, j, b), tmp_score);
        if (tmp_score > tmp_max_score) {
            tmp_max_score = tmp_score;
            local_best_seqs = tmp_seqs;
        }
    }

    playout_start->add_ts_score(tmp_max_score, 1);

    if (tmp_max_score > max_score) {
        max_mut.lock();
        if (tmp_max_score > max_score) {
            max_score = tmp_max_score;
            if (max_score == 0) print_arr_gkk(local_best_seqs);
        }
        max_mut.unlock();
    }
}

int main() {
    std::mt19937 mt(34943);
    std::uniform_int_distribution<int> dist(0, inf);

    ARR_GKK seqs = {};
    ARR_GG pair_count = {};

    int perm[g-1];
    for (int z = 0; z < g; z++) {
        std::iota(perm, perm + g-1, 0);
        for (int i = 0; i < g-1; i++) if (perm[i] >= z) perm[i]++;
        std::shuffle(perm, perm + g-1, mt);

        for (int i = 0, j = 0; i < k/2; i++, j += k/2) {
            std::copy(perm + j, perm + j + k/2, seqs[z][i].begin());
        }

        for (int i = 0; i < k/2; i++) {
            for (int j = 0; j < k/2; j++) {
                for (int t = j+1; t < k/2; t++) {
                    pair_count[std::min(seqs[z][i][j], seqs[z][i][t])][std::max(seqs[z][i][j], seqs[z][i][t])]++;
                }
            }
        }
    }

    State *root = new State;
    root->score = 0;
    for (int i = 0; i < g; i++) {
        for (int j = i+1; j < g; j++) root->score -= abs(pair_count[i][j] - expected);
    }

    int max_score = root->score, cnt_resets = 0, max_depth = 0;
    std::mutex max_mut;
    while (max_score < 0) {
        cnt_resets++;

        ARR_GKK tmp_seqs = seqs;
        ARR_GG tmp_pair_count = pair_count;

        std::vector<State *> states = {root};
        while ((int)root->children_transforms.size() == max_num_children) {
            Transform best_trans;
            double best_fit = -inf;

            for (const auto &[trans, child]: root->children_transforms) {
                double child_fit = child->fitness(root->times_visited);
                if (best_fit < child_fit) {
                    best_trans = trans;
                    best_fit = child_fit;
                }
            }

            transform_sequences(tmp_seqs, tmp_pair_count, best_trans, root->score);
            root = root->children_transforms[best_trans];
            states.push_back(root);
        }

        std::vector<std::pair<Transform, State *>> playout_starts;
        for (int z = 0; z < g; z++)
            for (int i = 0; i < k/2; i++)
                for (int a = 0; a < k/2; a++)
                    for (int j = i+1; j < k/2; j++)
                        for (int b = 0; b < k/2; b++) {
                            Transform curr_trans = Transform(z, i, a, j, b);

                            if (root->children_transforms.find(curr_trans) == root->children_transforms.end()) {
                                State *playout_start = new State;
                                root->children_transforms[curr_trans] = playout_start;
                                playout_starts.emplace_back(curr_trans, playout_start);
                            }
                        }

        std::vector<std::thread> threads;
        for (const auto &[trans, playout_start]: playout_starts) {
            threads.push_back(std::thread(playout_worker, dist(mt), std::ref(max_score), std::ref(max_mut),
                                          root->score, playout_start, trans, tmp_seqs, tmp_pair_count));
        }

        for (std::thread &th: threads) th.join();

        int playout_ts_score_sum = 0;
        //std::vector<double> playout_ts_fits;
        for (const auto &[_, playout_start]: playout_starts) {
            playout_ts_score_sum += playout_start->ts_score;
            //playout_ts_fits.push_back((double)playout_start->ts_score / -mean_random_scores[(k-4)/2] + 1.0);
        }

        //double dbg_mean = (double)std::accumulate(playout_ts_fits.begin(), playout_ts_fits.end(), 0.0) / threads.size();
        //double dbg_std = 0;
        //for (double fit: playout_ts_fits) dbg_std += (fit - dbg_mean) * (fit - dbg_mean);
        //dbg_std = sqrt(dbg_std / (playout_ts_fits.size() - 1));

        //std::cout << std::fixed << std::setprecision(3) << "playout scores mean = " << dbg_mean << ", ";
        //std::cout << std::fixed << std::setprecision(3) << "std = " << dbg_std << ", ";
        //std::cout << "min = " << *std::min_element(playout_ts_fits.begin(), playout_ts_fits.end()) << ", ";
        //std::cout << "max = " << *std::max_element(playout_ts_fits.begin(), playout_ts_fits.end()) << '\n';

        max_depth = std::max(max_depth, (int)states.size() + 1);

        root = states[0];
        while (states.size()) {
            states.back()->add_ts_score(playout_ts_score_sum, threads.size());
            states.pop_back();
        }

        std::cout << "iter = " << cnt_resets << ", max_depth = " << max_depth << ", max_score = " << max_score << '\n';
        // std::cout << std::fixed << std::setprecision(3) << ", root avg score = " << root->ts_score / root->times_visited << '\n';

        //aaa
    }

    return 0;
}
