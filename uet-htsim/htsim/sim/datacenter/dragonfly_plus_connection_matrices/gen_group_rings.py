#!/usr/bin/env python

# Generate a group ring traffic matrix.
# E.g.
# Host 0 Group 0 sends to Host 0 Group 1
# Host 0 Group 1 sends to Host 0 Group 2
# ...
# Host 0 Group n sends to Host 0 Group 0
#
# The number of active hosts per group is defaulted to 8
#
# python gen_ring.py <nodes> <flowsize> <active> <extrastarttime>
# Parameters:
# <nodes>   number of nodes in the topology
# <flowsize>   size of the flows in bytes
# <active>      number of active hosts per group
# <extrastarttime>   How long in microseconds to space the start times over (start time will be random in between 0 and this time).  Can be a float.

import os
import sys
from random import seed, shuffle
#print(sys.argv)

if len(sys.argv) == 1:
    print("Using default parameters from script")
    print("For command line parameters use: python gen_ring.py <filename> <nodes> <flowsize> <extrastarttime>")
    filename = "ring_256_4352n_2MB.cm"
    nodes = 4352
    flowsize = 2000000
    active = 256
    extrastarttime = 0
    randseed = 0
elif len(sys.argv) != 5:
    print("Usage: python gen_ring.py <filename> <nodes> <flowsize> <extrastarttime>")
    sys.exit()
else:
    filename = sys.argv[1]
    nodes = int(sys.argv[2])
    flowsize = int(sys.argv[3])
    extrastarttime = float(sys.argv[4])

groups = 17
nodes_per_group = nodes // groups

print("Nodes: ", nodes)
print("Flowsize: ", flowsize, "bytes")
print("ExtraStartTime: ", extrastarttime, "us")

srcs = []
dsts = []

for group in range(groups):
    for i in range(active):
        src = group * nodes_per_group + i
        dst = (group + 1) * nodes_per_group + i
        if group == groups - 1:
            dst = i
        srcs.append(src)
        dsts.append(dst)

conns = len(srcs)

f = open(filename, "w")
print("Nodes", nodes, file=f)
print("Connections", conns, file=f)

for n in range(conns):
    out = str(srcs[n]) + "->" + str(dsts[n]) + " id " + str(n+1) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
    print(out, file=f)

f.close()
