import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


def generate_parallel_tm(args, use_groups: list, tm_file: str):
    cnt_nodes = len(use_groups) * args.GROUP_SIZE
    use_nodes = [i for g in use_groups for i in range(g * args.GROUP_SIZE, (g + 1) * args.GROUP_SIZE)]

    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {cnt_nodes * (cnt_nodes - 1)}\n")

        flow_id = 1
        for i in use_nodes:
            for j in use_nodes:
                if j != i:
                    fout.write(f"{i}->{j} id {flow_id} start {0} size {args.FLOW_SIZE}\n")
                    flow_id += 1


def run_sim(args, topos: list, bibd_file: str):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    tm_files = []
    with open(bibd_file) as fin:
        for i, line in enumerate(fin.readlines()):
            tm_files.append(os.path.join(args.TM_FOLDER, f"experiment_{du.RUN_ID}_{i}_tmp.cm"))
            generate_parallel_tm(args, use_groups = list(map(int, line.strip().split())), tm_file = tm_files[-1])

    t_start = time.time()
    for topo in topos:
        cmds = [
            du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{nt}_{id_tmf}.dat")
            for id_tmf, tm_file in enumerate(tm_files) for nt in range(args.CNT_RUNS_PER_TOPO)
        ]

        srs.extend(pool.map(du.proc_run, cmds))

        print(f"Finished {topo = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    for tm_file in tm_files:
        os.system(f"rm {tm_file}")

    return srs


def main(args):
    for root, dir, files in os.walk(os.path.join(args.TM_FOLDER, "partial_all_to_all_BIBDs")):
        for file in files:
            if file.startswith("BIBD") and file.endswith(".txt") and int(file.split("n_")[1].split('_')[0]) == args.CNT_GROUPS:
                cnt_used_groups = int(file.split("k_")[1].split('_')[0])

                print(f"dbg {file = }, {args.K = }, {args.CNT_GROUPS = }, {cnt_used_groups = }")

                ht = du.get_default_ht(args)
                ht["EXP_TYPE"] = "partial_all_to_all"
                ht["BIBD_file"] = file

                t_start = time.time()

                for topo_name in args.TOPOS[args.K]:
                    srs = run_sim(args, topos = args.TOPOS[args.K][topo_name], bibd_file = os.path.join(root, file))

                    ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
                    
                    print(f"Finished {topo_name = } with {cnt_used_groups = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

                ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

                with open(f"./jsons/partial_all_to_all_{int(time.time())}.json", 'w') as fout:
                    json.dump(ht, fout, indent = 4)

                du.post_run_cleanup(args)

                print(f"Finished {cnt_used_groups = }/{args.CNT_GROUPS}. {round(time.time() - t_start, 3)} s passed.", flush = True)


if __name__ == "__main__":
    parser = du.get_default_parser()

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")

    main(args)
