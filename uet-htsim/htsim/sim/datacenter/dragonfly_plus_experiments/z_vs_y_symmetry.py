from subprocess import Popen, PIPE
import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import math
import os
import re

import dfp_exp_utils as du


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "z_vs_y_symmetry"
    ht["BEST_Z_TOPO"] = args.BEST_Z_TOPO
    ht["WORST_Z_TOPO"] = args.WORST_Z_TOPO

    t_start = time.time()

    pool = mp.Pool(processes = du.CNT_PROCESSES)

    cmds = [
        du.get_htsim_cmdlist(args, args.SEEDS[nt], args.TM_FILE, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat")
        for tid, topo in enumerate([args.BEST_Z_TOPO, args.WORST_Z_TOPO]) for nt in range(args.CNT_RUNS_PER_TOPO)
    ]

    srs = pool.map(du.proc_run, cmds)

    ht["results_best"] = [srs[i].fcts[-1] for i in range(args.CNT_RUNS_PER_TOPO)]
    ht["results_worst"] = [srs[i].fcts[-1] for i in range(args.CNT_RUNS_PER_TOPO, 2 * args.CNT_RUNS_PER_TOPO)]

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/z_vs_y_symmetry_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--BEST_Z_TOPO", type = str,
        help = "(specific to z_vs_y_symmetry.py) filepath to a topology with a good Z-symmetry score."
    )

    parser.add_argument(
        "--WORST_Z_TOPO", type = str,
        help = "(specific to z_vs_y_symmetry.py) filepath to a topology with a worse Z-symmetry score."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..

    main(args)
