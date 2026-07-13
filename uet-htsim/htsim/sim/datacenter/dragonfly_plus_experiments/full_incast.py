import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


def generate_transport_matrix(args, incasted_host: int):
    with open(args.TM_FILE, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {args.CNT_NODES - 1 if args.INCAST_TYPE == 'host' else args.CNT_NODES - args.GROUP_SIZE}\n")

        j = 1
        if args.INCAST_TYPE == "host":
            for i in range(args.CNT_NODES):
                if i != incasted_host:
                    fout.write(f"{i}->{incasted_host} id {j} start {0} size {args.FLOW_SIZE}\n")
                    j += 1
        else: # group incast.
            incasted_host -= incasted_host % args.GROUP_SIZE
            to = incasted_host
            for i in range(args.CNT_NODES):
                if i // args.GROUP_SIZE != incasted_host // args.GROUP_SIZE:
                    fout.write(f"{i}->{to} id {j} start {0} size {args.FLOW_SIZE}\n")
                    to = to + 1 if to + 1 < incasted_host + args.GROUP_SIZE else incasted_host
                    j += 1


def run_sim(args, topos: list):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    for incasted_host in range(0, args.CNT_NODES, args.GROUP_SIZE):
        generate_transport_matrix(args, incasted_host)

        cmds = [
            du.get_htsim_cmdlist(args, args.SEEDS[nt], args.TM_FILE, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat")
            for tid, topo in enumerate(topos) for nt in range(args.CNT_RUNS_PER_TOPO)
        ]

        srs.extend(pool.map(du.proc_run, cmds))

        print(f"Finished group {incasted_host // args.H**2} / {args.CNT_GROUPS}. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "full_incast"
    ht["INCAST_TYPE"] = args.INCAST_TYPE
    ht["HT_FCT_KEEP"] = args.HT_FCT_KEEP

    t_start = time.time()
    for topo_name in args.TOPOS[args.K]:
        srs = run_sim(args, topos = args.TOPOS[args.K][topo_name])
        
        if args.HT_FCT_KEEP == "mean":
            ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        else:
            ht[topo_name] = {"fcts": [sr.fcts for sr in srs], "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}

        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_incast_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--INCAST_TYPE", type = str, default = "group",
        choices = ["group", "host"],
        help = "(specific to full_incast.py) type of incast: all-to one host, all-to one group."
    )

    parser.add_argument(
        "--HT_FCT_KEEP", type = str, default = "mean",
        choices = ["mean", "all"],
        help = "(specific to full_incast.py) how to keep FCTs, mean over all measurements"
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    main(args)
