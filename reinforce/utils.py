import torch.nn.functional as F
import numpy as np
import random
import torch
import numba
import math

K = 4
G = K**2 // 4 + 1
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

    for z in range(G):
        for i in range(G):
            if i != z:
                score -= np.abs(X[:, z, i].sum(axis = -1) - EXPECTED).astype(np.int32)

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

