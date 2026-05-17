import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


def run_sim(topos: list, tm_file: str):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    
    cmds = [
        du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat")
        for tid, topo in enumerate(topos) for nt in range(args.CNT_RUNS_PER_TOPO)
    ]

    srs = pool.map(du.proc_run, cmds)

    print(f"Finished {topos[0] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "full_all_to_all"
    ht["ALL_TO_ALL_TYPE"] = args.ALL_TO_ALL_TYPE

    t_start = time.time()
    for topo_name in args.TOPOS[args.K]:
        tm_file = os.path.join(args.TM_FOLDER, f"parallel_all_to_all_k_{args.K}_2MB.cm")

        srs = run_sim(topos = args.TOPOS[args.K][topo_name], tm_file = tm_file)

        cnt_sims = len(srs)
        if args.ALL_TO_ALL_TYPE == "parallel_pfc":
            with open(tm_file) as fin:
                fin.readline()
                cnt_flows = int(fin.readline().strip().split()[-1])

            cnt_failed_sims = sum([len(sr.fcts) < cnt_flows for sr in srs])
            srs = [sr for sr in srs if len(sr.fcts) == cnt_flows]
        else:
            cnt_failed_sims = 0

        ht[topo_name] = {
            "mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist() if len(srs) else [],
            "mean_rtx": np.array([sr.rtx for sr in srs]).mean() if len(srs) else [],
            "rap_failed_sims": cnt_failed_sims / cnt_sims,
        }

        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--ALL_TO_ALL_TYPE", type = str, default = "parallel",
        choices = ["parallel", "parallel_pfc"],
        help = "(specific to full_all_to_all.py) type of run."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")

    main(args)
