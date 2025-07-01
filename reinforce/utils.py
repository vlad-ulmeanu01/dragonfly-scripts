import torch.nn.functional as F
import numpy as np
import torch
import math

K = 6
G = K**2 // 4 + 1
EXPECTED = K // 2 - 1
inf = (1 << 30)

MAX_EPISODE_LEN = 100
NUM_EPISODES = 10 ** 3
BATCH_SIZE = 1024
TEMP = 1.0

SEED = 34989348

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
