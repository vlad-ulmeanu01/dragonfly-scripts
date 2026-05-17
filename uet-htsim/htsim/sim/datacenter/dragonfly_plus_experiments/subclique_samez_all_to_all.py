import multiprocessing as mp
import numpy as np
import itertools
import random
import time
import json
import os

import dfp_exp_utils as du


def generate_subclique_tm(args, clique_gs: list, tm_file: str):
    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {len(clique_gs) * (len(clique_gs) - 1) * args.GROUP_SIZE**2}\n")

        conn_id = 1
        for i, j in itertools.product(clique_gs, clique_gs):
            if i != j:
                for x, y in itertools.product(range(i * args.GROUP_SIZE, (i+1) * args.GROUP_SIZE), range(j * args.GROUP_SIZE, (j+1) * args.GROUP_SIZE)):
                    fout.write(f"{x}->{y} id {conn_id} start {0} size {args.FLOW_SIZE}\n")
                    conn_id += 1


def find_subcliques(args, topos: list):
    ht_tm = {} # ht_tm[z_same][len clique] = (topo = config file, tm_file).

    tm_id = 0
    seen_none = False
    for topo in topos:
        if topo is not None or not seen_none:
            Z = [[0 for j in range(args.CNT_GROUPS)] for i in range(args.CNT_GROUPS)]

            if topo is None:
                seen_none = True
                rows = [[[] for j in range(args.H)] for i in range(args.CNT_GROUPS)]
                for i in range(args.CNT_GROUPS):
                    for j in range(i+1, args.CNT_GROUPS):
                        src_sw, dst_sw = (j - 1) // args.H, i // args.H # src_sw, dst_sw are spine switch ids, local for each group.
                        rows[i][src_sw].append(j)
                        rows[j][dst_sw].append(i)
            else:
                rows = []
                with open(topo) as fin:
                    for line in fin.readlines():
                        row = list(map(int, line.strip().split()))
                        rows.append([row[i: i+args.H] for i in range(0, len(row), args.H)])

            for row in rows:
                for buck in row:
                    for i in range(args.H):
                        for j in range(i+1, args.H):
                            Z[min(buck[i], buck[j])][max(buck[i], buck[j])] += 1

            for mask in range(1 << args.CNT_GROUPS):
                clique_gs = [i for i in range(args.CNT_GROUPS) if mask & (1 << i)]
                zs = [Z[clique_gs[i]][clique_gs[j]] for i in range(len(clique_gs)) for j in range(i+1, len(clique_gs))]
                if zs and min(zs) == max(zs):
                    z_same = zs[0]
                    if z_same not in ht_tm:
                        ht_tm[z_same] = {}
                    if len(clique_gs) not in ht_tm[z_same]:
                        ht_tm[z_same][len(clique_gs)] = []
                    
                    tm_file = os.path.join(args.TM_FOLDER, f"experiment_{du.RUN_ID}_{tm_id}.cm")
                    generate_subclique_tm(args, clique_gs, tm_file)
                    ht_tm[z_same][len(clique_gs)].append((topo, tm_file))
                    tm_id += 1

    return ht_tm


def run_sim(args, arr_topos_tm_files: list):
    if len(arr_topos_tm_files) > args.MAX_SUBCLIQUES_PER_RUN_SIM:
        random.seed(args.SUBCLIQUE_SHUFFLE_SEED)
        random.shuffle(arr_topos_tm_files)
        arr_topos_tm_files = arr_topos_tm_files[: args.MAX_SUBCLIQUES_PER_RUN_SIM]

    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    cmds = [
        du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{nt}_{tid}.dat")
        for nt in range(args.CNT_RUNS_PER_TOPO) for tid, (topo, tm_file) in enumerate(arr_topos_tm_files)
    ]

    srs = pool.map(du.proc_run, cmds)

    return srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "subclique_samez_all_to_all"
    ht["MAX_SUBCLIQUES_PER_RUN_SIM"] = args.MAX_SUBCLIQUES_PER_RUN_SIM
    ht["SUBCLIQUE_SHUFFLE_SEED"] = args.SUBCLIQUE_SHUFFLE_SEED

    t_start = time.time()

    ht_tm = find_subcliques(args, topos = [topo for topo_name in args.TOPOS[args.K] for topo in args.TOPOS[args.K][topo_name]])

    for z_same in ht_tm:
        ht[f"z_same_{z_same}"] = {}
        for len_clique in ht_tm[z_same]:
            srs = run_sim(args, ht_tm[z_same][len_clique])
            ht[f"z_same_{z_same}"][f"len_clique_{len_clique}"] = {"fcts": [sr.fcts for sr in srs], "rtxs": [sr.rtx for sr in srs]}
            print(f"Finished {z_same = }, {len_clique = }: mean FCT = {np.mean(ht[f'z_same_{z_same}'][f'len_clique_{len_clique}']['fcts'])}. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/subclique_samez_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    for z_same in ht_tm:
        for len_clique in ht_tm[z_same]:
            for _, tm_file in ht_tm[z_same][len_clique]:
                os.system(f"rm {tm_file}")

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--MAX_SUBCLIQUES_PER_RUN_SIM", type = int, default = 10,
        help = "(specific to subclique_samez_all_to_all.py) We want to choose subcliques s.t. any x, y in the subcq have the same Z[x, y]. We simulate at most ?? subcqs with a fixed Z value."
    )

    parser.add_argument(
        "--SUBCLIQUE_SHUFFLE_SEED", type = int, default = 34893,
        help = "(specific to subclique_samez_all_to_all.py) If we find more than MAX_SUBCLIQUES_PER_RUN_SIM subcqs, we randomly select that many. Shuffle with this seed."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")

    main(args)
