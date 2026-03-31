from random import seed, shuffle, randint
import sys
import os

assert len(sys.argv) == 3, "Usage: python3 gen_full_incast.py <switch_radix> <flowsize (bytes)>"

k, flowsize = int(sys.argv[1]), int(sys.argv[2])
assert k >= 4, "Need switch_radix >= 4"
assert flowsize > 0, "Need flowsize(bytes) > 0"

print(f"switch_radix = {k}, flow size = {flowsize}")

fs_human = f"{flowsize}B" if flowsize < 10**3 else (f"{flowsize // 10**3}KB" if flowsize < 10**6 else f"{flowsize // 10**6}MB")

h = k // 2
cnt_nodes = (h ** 2 + 1) * h**2

with open(f"full_incast_k_{k}_{fs_human}.cm", 'w') as fout:
    fout.write(f"Nodes {cnt_nodes}\n")
    fout.write(f"Connections {cnt_nodes - 1}\n")
    for i in range(1, cnt_nodes):
        fout.write(f"{i}->{0} id {i} start {0} size {flowsize}\n")
