"""
copy-paste formulele de la demonstratie fara vreo proprietate.
"""

from ortools.linear_solver import pywraplp
import itertools
import time
import sys
import os

ROOT_TOPO = "/home/vlad/Documents/SublimeMerge/dragonfly-scripts/simulate_dfly_queue_sizes/configs/"

def main():
    type_opt = [arg.split('=')[1] for arg in sys.argv if arg.startswith("type_opt=")][0]

    topo_name = [arg.split('=')[1] for arg in sys.argv if arg.startswith("topo_name=")][0]
    topo_name = os.path.join(ROOT_TOPO, topo_name)

    k = int(topo_name.split("k_")[1].split('/')[0])
    h = k // 2
    num_groups = h**2 + 1
    num_hosts = num_groups * h**2

    with open(topo_name) as fin:
        topo = [[row[i: i+h] for i in range(0, h**2, h)] for line in fin.readlines() for row in [list(map(int, line.strip().split()))]]

    solver = pywraplp.Solver.CreateSolver("GLOP")

    t_start = time.time()

    rate = [[solver.NumVar(0, 1, f"rate_{i}_{j}") for j in range(num_hosts)] for i in range(num_hosts)]

    print(f"alloc rate vars in {time.time() - t_start}")

    t_start = time.time()

    for i in range(num_hosts):
        solver.Add(sum([rate[i][j] for j in range(num_hosts)]) <= 1)
        solver.Add(sum([rate[j][i] for j in range(num_hosts)]) <= 1)
    
    print(f"add rate constraints in {time.time() - t_start}")

    t_start = time.time()

    if type_opt == "global":
        # global flow.
        group_from, group_to = 0, 1
        
        flow_xy = sum([ # x -> y direct
            rate[group_from * h**2 + i][group_to * h**2 + j]

            for i, j in itertools.product(range(h**2), range(h**2))
        ]) / h**2\
            +\
        sum([ # x -> y -> ?? (y intermediar 1/2-ind)
            rate[group_from * h**2 + i][group_oth * h**2 + j]

            for ind in range(h)
            for group_oth in topo[group_to][ind]
            if group_oth != group_from
            for i, j in itertools.product(range(h**2), range(h**2))
        ]) / h**2\
            +\
        sum([ # ?? -> x -> y (x intermediar 1/2-ind)
            rate[group_oth * h**2 + i][group_to * h**2 + j]

            for ind in range(h)
            for group_oth in topo[group_from][ind]
            if group_oth != group_to
            for i, j in itertools.product(range(h**2), range(h**2))
        ]) / h**2
    elif type_opt == "local":
        # local flow.
        group_id, leaf_id, spine_id = 0, 0, 0

        flow_xy = sum([ # LLLL
            rate[group_id * h**2 + leaf_id * h + i][group_id * h**2 + leaf_oth * h + j]

            for i in range(h)
            for leaf_oth in range(h) if leaf_oth != leaf_id
            for j in range(h)
        ]) / h\
            +\
        sum([ # LLGLL, LLGGLL, unele din LLGLLGLL (de fapt asta e cea mai grea parte. sum asta + cea de dinainte trb puse impreuna la bound)
            rate[group_id * h**2 + leaf_id * h + i][group_oth * h**2 + j]

            for e in topo[group_id][spine_id]
            for ind in range(h)
            for fg in topo[e][ind]
            for group_oth in [fg if fg != group_id else e]
            for i, j in itertools.product(range(h), range(h**2))
        ]) / h**2\
            +\
        sum([ # restul din LLGLLGLL
            rate[group_from * h**2 + i][group_to * h**2 + j]

            for group_from in topo[group_id][spine_id]
            for spine_oth in range(h) if spine_oth != spine_id
            for group_to in topo[group_id][spine_oth]
            for i, j in itertools.product(range(h**2), range(h**2))
        ]) / h**3
    else:
        assert(False)

    print(f"build rate_xy var in {time.time() - t_start}")

    print(f"Number of constraints = {solver.NumConstraints()}")
    t_start = time.time()

    solver.Maximize(flow_xy)
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print(f"Solver time: {time.time() - t_start}")
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value():0.1f}")
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()
