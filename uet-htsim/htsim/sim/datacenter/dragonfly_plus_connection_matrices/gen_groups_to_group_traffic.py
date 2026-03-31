#!/usr/bin/env python

# Generate a traffic matrix between hosts of the same id in different groups (e.g. first group to last group; first 2 groups to last group)
# python gen_groups_to_group_traffic.py <nodes> <conns> <flowsize> <extrastarttime>
# Parameters:
# <nodes>   number of nodes in the topology
# <conns>    number of active connections
# <flowsize>   size of the flows in bytes
# <extrastarttime>   How long in microseconds to space the start times over (start time will be random in between 0 and this time).  Can be a float.
# <randseed>   Seed for random number generator, or set to 0 for random seed

import os
import sys
from random import seed, shuffle
#print(sys.argv)

if len(sys.argv) == 1:
    print("Using default parameters from script")
    print("For command line parameters use: python gen_permutation_dragonfly_plus.py <filename> <nodes> <conns> <flowsize> <extrastarttime> <randseed>")
    filename = "2_1_4352n_2MB.cm"
    nodes = 4352
    conns = 512
    flowsize = 2000000
    extrastarttime = 0
    randseed = 0
elif len(sys.argv) != 7:
    print("Usage: python gen_permutation_dragonfly_plus.py <filename> <nodes> <conns> <flowsize> <extrastarttime> <randseed>")
    sys.exit()
else:
    filename = sys.argv[1]
    nodes = int(sys.argv[2])
    conns = int(sys.argv[3])
    flowsize = int(sys.argv[4])
    extrastarttime = float(sys.argv[5])
    randseed = int(sys.argv[6])

print("Nodes: ", nodes)
print("Connections: ", conns)
print("Flowsize: ", flowsize, "bytes")
print("ExtraStartTime: ", extrastarttime, "us")
print("Random Seed ", randseed)

f = open(filename, "w")
print("Nodes", nodes, file=f)
print("Connections", conns, file=f)

srcs = []
dsts = []
for n in range(conns):
    srcs.append(n)
    dsts.append(n % 256 + 4096)
if randseed != 0:
    seed(randseed)
#print(srcs)
#print(dsts)

for n in range(conns):
    out = str(srcs[n]) + "->" + str(dsts[n]) + " id " + str(n + 1) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
    print(out, file=f)

f.close()
