#!/usr/bin/env python

# Generate a ring traffic matrix.
# python gen_ring.py <nodes> <flowsize> <extrastarttime>
# Parameters:
# <nodes>   number of nodes in the topology
# <flowsize>   size of the flows in bytes
# <extrastarttime>   How long in microseconds to space the start times over (start time will be random in between 0 and this time).  Can be a float.

import os
import sys
from random import seed, shuffle
#print(sys.argv)

if len(sys.argv) == 1:
    print("Using default parameters from script")
    print("For command line parameters use: python gen_ring.py <filename> <nodes> <flowsize> <extrastarttime>")
    filename = "ring_4352n_2MB.cm"
    nodes = 4352
    flowsize = 2000000
    extrastarttime = 0
    randseed = 0
    random_ring = False
elif len(sys.argv) != 5:
    print("Usage: python gen_ring.py <filename> <nodes> <flowsize> <extrastarttime>")
    sys.exit()
else:
    filename = sys.argv[1]
    nodes = int(sys.argv[2])
    flowsize = int(sys.argv[3])
    extrastarttime = float(sys.argv[4])

conns = nodes

print("Nodes: ", nodes)
print("Flowsize: ", flowsize, "bytes")
print("ExtraStartTime: ", extrastarttime, "us")

f = open(filename, "w")
print("Nodes", nodes, file=f)
print("Connections", conns, file=f)

srcs = []
dsts = []
for n in range(nodes):
    srcs.append(n)
    dsts.append(n + 1)
dsts[-1] = 0
if random_ring:
    shuffle(srcs)
    for n in range(conns - 1):
        out = str(srcs[n]) + "->" + str(srcs[n + 1]) + " id " + str(n+1) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
        print(out, file=f)
    out = str(srcs[n + 1]) + "->" + str(srcs[0]) + " id " + str(n+2) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
    print(out, file=f)
else:
    for n in range(conns):
        out = str(srcs[n]) + "->" + str(dsts[n]) + " id " + str(n+1) + " start " + str(int(extrastarttime * 1000000)) + " size " + str(flowsize)
        print(out, file=f)

f.close()
