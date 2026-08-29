import random

cnt_runs = 1000

A, B = map(int, input().split())
cnt_over = []

for _ in range(cnt_runs):
    bins = [0] * B
    for z in range(A):
        i, j = random.randrange(B), random.randrange(B)
        if bins[i] > bins[j]:
            i = j
        bins[i] += 1
    cnt_over.append(sum([bins[i] > A // B for i in range(B)]))

print(sum(cnt_over) / cnt_runs / B)
