import numpy as np
import random
import math

print(f"n, nt = ", end = '')
n, nt = map(int, input().split())

freq = [0] * (n + 1)

for _ in range(nt):
    arr = [i for i in range(n) for j in range(n)]
    random.shuffle(arr)

    cnt_0s_first_group = sum([arr[i] == 0 for i in range(n)])

    freq[cnt_0s_first_group] += 1

freq = np.array(freq) / nt
print(f"freq = {freq.round(3)} (freq[0] ar tb {math.exp(-1)}), E(freq) = {sum(freq * np.arange(len(freq)))} (1 ar tb sa dea)")
