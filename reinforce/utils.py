K = 6
G = K**2 // 4 + 1
EXPECTED = K // 2 - 1
inf = (1 << 30)

MAX_EPISODE_LEN = 500
NUM_EPISODES = 10 ** 4
BATCH_SIZE = 128
TEMP = 1.0

SEED = 34989348


def print_debug_ht(ht):
    print(', '.join([f"{s}: {round(ht[s], 3)}" for s in ht]), flush = True)
