import numpy as np

K = 6
G = K**2 // 4 + 1
EXPECTED = K // 2 - 1
inf = (1 << 30)

MAX_EPISODE_LEN = 100
NUM_EPISODES = 10 ** 3
BATCH_SIZE = 128
TEMP = 1.0

SEED = 34989348


def print_debug_ht(ht):
    print(', '.join([f"{s}: {round(ht[s], 3)}" for s in ht]), flush = True)

def get_state_from_X(X: np.array):
    state = [[[] for _ in range(K // 2)] for _ in range(G)]
    
    for z in range(G):
        seen = [False] * G
        seen[z] = True
        
        curr_group = -1
        for i in range(G):
            if not seen[i]:
                curr_group += 1
                state[z][curr_group].append(i+1)
                seen[i] = True
                for j in range(G):
                    if X[z, i, j]:
                        state[z][curr_group].append(j+1)
                        seen[j] = True
                        
    return state
