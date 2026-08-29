"""
We have N buckets, each with N slots each. The first bucket has N 1s, the second N 2s, the last N Ns. We shuffle all slots in all buckets.
What is the expected number of distinct numbers in each bucket? Lower bound whp? Expected without number that already was in that bucket?

Easier: N 1s, rest have 0s. Shuffle. Expected number of buckets with at least a 1.
"""

import random


def mean_number_of_buckets_with_ones(n, nt):
    arr = [True] * n + [False] * (n**2 - n)

    cnt = 0
    for _ in range(nt):
        random.shuffle(arr)
        cnt += sum([any(arr[i: i+n]) for i in range(0, n**2, n)])

    print(f"mean number of buckets with 1s: {round(cnt/nt, 3)} out of {n}")


def mean_number_of_distinct_numbers_per_bucket(n, nt):
    arr = [j for i in range(1, n+1) for j in [i] * n]

    cnt = 0
    for _ in range(nt):
        random.shuffle(arr)
        cnt += sum([len(set(arr[i: i+n])) for i in range(0, n**2, n)]) / n

    print(f"mean number of distinct numbers per bucket: {round(cnt/nt, 3)} out of {n}")


def main():
    print(f"n, nt = ", end = '', flush = True)
    n, nt = map(int, input().split())

    mean_number_of_buckets_with_ones(n, nt)
    mean_number_of_distinct_numbers_per_bucket(n, nt)


if __name__ == "__main__":
    main()