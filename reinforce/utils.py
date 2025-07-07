import torch.nn.functional as F
import numpy as np
import random
import torch
import numba
import math
import copy

K = 6
G = K**2 // 4 + 1
E_MEAN_SCORE, E_STD_SCORE = -40, 5.8
EXPECTED = K // 2 - 1
inf = (1 << 30)

MAX_EPISODE_LEN = 100
NUM_EPISODES = 10 ** 4
BATCH_SIZE = 30
TEMP = 1.0


SAVE_EVERY = 100

SEED = 34989348

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



def generate_new_start(batch_size: int):
    X = np.zeros((batch_size, G, G, G), dtype = np.float32)

    for bid in range(batch_size):
        group_ids = list(range(G))
        for z in range(G):
            random.shuffle(group_ids)
            for group in np.array([i for i in group_ids if i != z]).reshape(K // 2, K // 2):
                for i in range(K // 2):
                    for j in range(i+1, K // 2):
                        m, M = min(group[i], group[j]), max(group[i], group[j])
                        X[bid, z, m, M] = X[bid, z, M, m] = 1.0

    return torch.from_numpy(X)


@numba.jit(nopython = True)
def evaluate_state(X: np.array):
    score = np.zeros(len(X), dtype = np.int32) # len(X) = batch_size.

    for i in range(G):
        for j in range(i+1, G):
            score -= np.abs(X[:, :, i, j].sum(axis = -1) - EXPECTED).astype(np.int32)

    return score


@numba.jit(nopython = True)
def do_move(X: np.array, actions: np.array):
    for bid, (z, i, j) in zip(range(len(X)), actions):
        if X[bid, z, i, j] == 0:
            i_buck_inds = np.arange(G)[X[bid, z, i, :] == 1.0]
            j_buck_inds = np.arange(G)[X[bid, z, :, j] == 1.0]

            X[bid, z, i, i_buck_inds] = 0.0; X[bid, z, i_buck_inds, i] = 0.0
            X[bid, z, j_buck_inds, j] = 0.0; X[bid, z, j, j_buck_inds] = 0.0

            X[bid, z, j, i_buck_inds] = 1.0; X[bid, z, i_buck_inds, j] = 1.0
            X[bid, z, j_buck_inds, i] = 1.0; X[bid, z, i, j_buck_inds] = 1.0

    return X


# daca apelam functia asta, policy net alege doar ce numere sa intoarca. noi alegem (stocastic) greedy linia pe care le interschimbam, e.g. calculam arr_z.
def greedy_sample_rows(X: np.array, arr_i: np.array, arr_j: np.array):
    og_X = copy.deepcopy(X)
    scores = []

    for z in range(G):
        X = do_move(X, np.array([(z, i, j) for i, j in zip(arr_i, arr_j)]))
        # scores.append(np.where(np.array([z == i or z == j for i, j in zip(arr_i, arr_j)]), -inf, evaluate_state(X)))
        scores.append(np.where(np.array([z == i or z == j for i, j in zip(arr_i, arr_j)]), -inf, evaluate_state(X)))

        del X
        X = copy.deepcopy(og_X)

    return F.softmax(torch.from_numpy(np.vstack(scores).T).float(), dim = 1).multinomial(num_samples = 1).flatten().numpy()


@numba.jit(nopython = True)
def compute_expert_actions(X: np.array, mask_expert_choices: np.array):
    expert_inds = np.zeros(len(X), dtype = np.int32) + (G + 2) # pot sa existe (pe langa starile unde nu vrem expert) stari care sunt maxim local. G + 2 = (0, 1, 2).

    for bid in range(len(X)):
        if mask_expert_choices[bid]:
            worst_diff, worst_ij = X[bid, :, 1, 2].sum() - EXPECTED, (1, 2)
            for i in range(G):
                for j in range(i+1, G):
                    diff = X[bid, :, i, j].sum() - EXPECTED
                    if abs(diff) > abs(worst_diff):
                        worst_diff, worst_ij = diff, (i, j)

            i, j = worst_ij
            if worst_diff < 0:
                z = np.random.choice(np.array([z for z in range(G) if X[bid, z, i, j] == 0]))
                expert_inds[bid] = z * G**2 + i * G + j
            elif worst_diff > 0:
                z = np.random.choice(np.array([z for z in range(G) if X[bid, z, i, j] == 1]))
                flip_i_with = np.random.choice(np.array([j for j in range(G) if X[bid, z, i, j] == 0 and j != i]))
                expert_inds[bid] = z * G**2 + i * G + flip_i_with

    return expert_inds


def get_state_from_X(X: np.array):
    state = [[] for _ in range(G)]

    for z in range(G):
        seen = [False] * G
        seen[z] = True

        curr_group = -1
        for i in range(G):
            if not seen[i]:
                state[z].append([i+1])
                seen[i] = True
                for j in range(G):
                    if X[z, i, j]:
                        state[z][-1].append(j+1)
                        seen[j] = True

    return state


class DecayScheduler:
    def __init__(self, start: float, end: float, decay: float):
        self.start = start
        self.end = end
        self.decay = decay

    def get(self, episode_ind: int):
        amt = math.exp(-episode_ind / self.decay)
        if self.end < self.start:
            return self.end + (self.start - self.end) * amt
        return self.start + (self.end - self.start) * (1 - amt)


def print_debug_ht(ht):
    print(', '.join([f"{s}: {round(ht[s], 3)}" for s in ht]), flush = True)


def proba_most_std_away(std: float):
    return 0.5 * (math.erf(std * 0.7071067811865476) - math.erf(-std * 0.7071067811865476))
