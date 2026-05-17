import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import os

import dfp_exp_utils as du


def run_sim(args, topos: list, tm_files: list):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    for topo in topos:
        cmds = [
            du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{tmf_id}_{nt}.dat")
            for tmf_id, tm_file in enumerate(tm_files) for nt in range(args.CNT_RUNS_PER_TOPO)
        ]

        srs.extend(pool.map(du.proc_run, cmds))

        print(f"Finished {topo = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "full_permutation"
    ht["HT_TYPE"] = args.HT_TYPE

    tm_files = [os.path.join(root, file) for root, dir, files in os.walk(args.PERM_TM_FOLDER) for file in files]
    
    t_start = time.time()
    for topo_name in args.TOPOS[args.K]:
        srs = run_sim(args, topos = args.TOPOS[args.K][topo_name], tm_files = tm_files)

        if args.HT_TYPE == "standard":
            ht[topo_name] = {"sorted_fcts": sorted([sr.fcts[-1] for sr in srs]), "sorted_rtxs": sorted([sr.rtx for sr in srs])}
        else:
            i = 0
            ht[topo_name] = {}
            for topo, tm_file in itertools.product(args.TOPOS[args.K][topo_name], tm_files):
                ht[topo_name][f"{topo}, {tm_file}"] = np.mean([sr.fcts[-1] for sr in srs[i: i + args.CNT_RUNS_PER_TOPO]])
                i += args.CNT_RUNS_PER_TOPO

        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_permutation_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--HT_TYPE", type = str, default = "config_tm_pair",
        choices = ["standard", "config_tm_pair"],
        help = "(specific to full_permutation.py) json ht output format."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.PERM_TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/full_permutations/k_{args.K}")

    assert args.DO_CC != "no_cc_pfc", "DO_CC = no_cc_pfc not allowed for full_permutation"

    main(args)
