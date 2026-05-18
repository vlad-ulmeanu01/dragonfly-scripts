from subprocess import Popen, PIPE
import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import os
import re

import dfp_exp_utils as du


# Flow Uec_0_16 flowId 1 uecSrc 0 starting at 0
# Flow Uec_1_0 flowId 2 uecSrc 1 starting at 0
# Flow Uec_2_15 flowId 3 uecSrc 2 starting at 0
# Flow Uec_3_5 flowId 4 uecSrc 3 starting at 0
# Flow Uec_4_7 flowId 5 uecSrc 4 starting at 0
# Flow Uec_5_12 flowId 6 uecSrc 5 starting at 0
# Flow Uec_6_2 flowId 7 uecSrc 6 starting at 0
# Flow Uec_7_1 flowId 8 uecSrc 7 starting at 0
# Flow Uec_8_14 flowId 9 uecSrc 8 starting at 0
# Flow Uec_9_10 flowId 10 uecSrc 9 starting at 0
# Flow Uec_10_11 flowId 11 uecSrc 10 starting at 0
# Flow Uec_11_17 flowId 12 uecSrc 11 starting at 0
# Flow Uec_12_3 flowId 13 uecSrc 12 starting at 0
# Flow Uec_13_19 flowId 14 uecSrc 13 starting at 0
# Flow Uec_14_4 flowId 15 uecSrc 14 starting at 0
# Flow Uec_15_9 flowId 16 uecSrc 15 starting at 0
# Flow Uec_16_18 flowId 17 uecSrc 16 starting at 0
# Flow Uec_17_13 flowId 18 uecSrc 17 starting at 0
# Flow Uec_18_6 flowId 19 uecSrc 18 starting at 0
# Flow Uec_19_8 flowId 20 uecSrc 19 starting at 0
# Flow Uec_1_0 flowId 2 uecSrc 1 finished at 166.903 total messages 1 total packets 490 RTS 0 total bytes 2002140 in_flight now 0 fair_inc 0 prop_inc 0 fast_inc 0 eta_inc 0 multi_dec -0 quick_dec -0 nack_dec -0


# if re.search(r"Flow .* flowId .* starting at", line):
#     host_from, host_to = map(int, line.split()[1].split('_')[1:])


def custom_proc_run(cmdlist):
    fcts_ext, rtx = [], None

    for line in Popen(cmdlist, shell = False, stdout = PIPE).stdout.readlines():
        line = line.decode().strip()

        if re.search(r"finished at (\d+)", line):
            fct = float(line.split("finished at ")[1].split(' ')[0])
            host_from, host_to = map(int, line.split()[1].split('_')[1:])
            fcts_ext.append((fct, host_from, host_to))

        if re.search(r"Rtx:", line):
            rtx = int(line.split("Rtx: ")[1].split(' ')[0])

    return du.SimResult(fcts_ext, rtx)


def generate_worse_tm_file(args, sr: du.SimResult, rap_w: float, tm_file: str):
    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {args.CNT_NODES * (args.CNT_NODES - 1)}\n")

        # aici in special sr.fcts tine tuplu (fct, host_from, host_to).
        last_fct = sr.fcts[-1][0]
        fcts = sorted(sr.fcts[: int(len(sr.fcts) * rap_w)], key = lambda p: (-p[1], -p[2]))

        z = 1
        for i in range(args.CNT_NODES):
            for j in range(args.CNT_NODES):
                if j != i:
                    flow_size = args.FLOW_SIZE
                    if fcts and (i, j) == fcts[-1][1:]:
                        fct = fcts[-1][0]
                        fcts.pop()

                        flow_size += int((last_fct - fct) * args.LINK_SPEED / 8)

                    fout.write(f"{i}->{j} id {z} start {0} size {flow_size}\n")
                    z += 1


def run_sim(args, topos: list, tm_file: str):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    ht_srs = {}

    t_start = time.time()
    
    cmds = [
        du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat")
        for tid, topo in enumerate(topos) for nt in range(args.CNT_RUNS_PER_TOPO)
    ]

    srs_base = pool.map(custom_proc_run, cmds)

    print(f"{topos[0] = }: finished srs_base. {round(time.time() - t_start, 3)} s passed.")

    for rap_w in args.WORSEN_RAPS:
        cmds = []
        for (i, sr), ((tid, topo), nt) in zip(enumerate(srs_base), itertools.product(enumerate(topos), range(args.CNT_RUNS_PER_TOPO))):
            tm_file = os.path.join(args.TM_FOLDER, f"tmp_worse_tm_file_{du.RUN_ID}_{tid}_{nt}.cm")
            generate_worse_tm_file(args, sr, rap_w, tm_file)
            cmds.append(du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat"))

        ht_srs[rap_w] = pool.map(du.proc_run, cmds)
        
        print(f"{topos[0] = }: finished {rap_w = }. {round(time.time() - t_start, 3)} s passed.")

    ht_srs[0.0] = [du.SimResult([fct for fct, _, _ in sr.fcts], sr.rtx) for sr in srs_base]

    print(f"Finished {topos[0] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return ht_srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "full_all_to_all_worsen_early_competions"
    ht["WORSEN_RAPS"] = args.WORSEN_RAPS

    t_start = time.time()
    for topo_name in args.TOPOS[args.K]:
        ht_srs = run_sim(
            args,
            topos = args.TOPOS[args.K][topo_name],
            tm_file = os.path.join(args.TM_FOLDER, f"parallel_all_to_all_k_{args.K}_2MB.cm")
        )

        for rap_w in ht_srs:
            ht[f"{topo_name}_{rap_w}"] = {
                "mean_fcts": np.array([sr.fcts for sr in ht_srs[rap_w]]).mean(axis = 0).tolist(),
                "mean_rtx": np.array([sr.rtx for sr in ht_srs[rap_w]]).mean()
            }

        print(f"Finished {topo_name = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_all_to_all_worsen_early_competions_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--WORSEN_RAPS", type = str, default = "[0.1,0.2,0.3,0.4,0.5]", # [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
        help = "(specific to full_all_to_all_worsen_early_competions.py) string array of floats (no spaces): the first ?% of finishing flows will be padded s.t. they finish at least at the actual FCT in a rerun."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")
    args.WORSEN_RAPS = list(map(float, args.WORSEN_RAPS.split('[')[-1].split(']')[0].split(',')))

    main(args)
