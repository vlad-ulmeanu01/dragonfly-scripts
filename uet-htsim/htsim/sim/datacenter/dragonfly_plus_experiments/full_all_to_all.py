# Generate connection matrices for ALL_TO_ALL_TYPE = "serial" with: python3 gen_serial_alltoall.py dfp_all_to_all_k_{K}.cm N N N 2000000 0 0
# Where N is the node count = H**2 * (H**2 + 1)

import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


# ALL_TO_ALL_TYPE = "serial"
# ALL_TO_ALL_TYPE = "parallel"
ALL_TO_ALL_TYPE = "parallel_pfc"
assert ALL_TO_ALL_TYPE in ["serial", "parallel", "parallel_pfc"]

K = 6

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = {"serial": 5, "parallel": 1, "parallel_pfc": 5}[ALL_TO_ALL_TYPE]
CNT_TMS_PER_TYPE = {"serial": 5, "parallel": 1, "parallel_pfc": 1}[ALL_TO_ALL_TYPE]
DO_CC = {"serial": du.DO_CC, "parallel": du.DO_CC, "parallel_pfc": "no_cc_pfc"}[ALL_TO_ALL_TYPE]
END_TIME = {"serial": du.END_TIME, "parallel": du.END_TIME, "parallel_pfc": du.END_TIME * 10}[ALL_TO_ALL_TYPE]

QUEUE_SIZE = {"serial": du.QUEUE_SIZE, "parallel": du.QUEUE_SIZE, "parallel_pfc": (K + 1) * du.BDP_PKTS}[ALL_TO_ALL_TYPE]
ECN = {"serial": du.ECN, "parallel": du.ECN, "parallel_pfc": (QUEUE_SIZE, QUEUE_SIZE)}[ALL_TO_ALL_TYPE]

TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


def run_sim(topos: list, tm_file: str):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    
    cmds = [
        du.get_htsim_cmdlist(
            seed = du.SEEDS[nt], tm_file = tm_file, end_time = END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = QUEUE_SIZE,
            ecn = ECN, topo = topo, do_cc = DO_CC, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat"
        )
        for tid, topo in enumerate(topos) for nt in range(CNT_RUNS_PER_TOPO)
    ]

    srs = pool.map(du.proc_run, cmds)

    print(f"Finished {topos[0] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main():
    ht = {
        "EXP_TYPE": "full_all_to_all", "ALL_TO_ALL_TYPE": ALL_TO_ALL_TYPE, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO,
        "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE, "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS,
        "DO_CC": DO_CC, "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": QUEUE_SIZE, "ECN": ECN,
        "TOTAL_TIME": 0
    }

    if ALL_TO_ALL_TYPE == "parallel_pfc":
        ht["PFC"] = (du.PFC_OFF, du.PFC_ON)
        ht["RTO"] = du.DFP_RTT + QUEUE_SIZE * du.DFP_MAX_HOPS

    t_start = time.time()
    for topo_name in du.TOPOS[K]:
        srs = []
        for tm_id in range(1, CNT_TMS_PER_TYPE + 1):
            tm_file = os.path.join(
                TM_FOLDER,
                f"dfp_all_to_all_k_{K}_no_{tm_id}.cm" if ALL_TO_ALL_TYPE == "serial" else f"parallel_all_to_all_k_{K}_2MB.cm"
            )
            srs.extend(run_sim(topos = du.TOPOS[K][topo_name], tm_file = tm_file))

        cnt_sims = len(srs)
        if ALL_TO_ALL_TYPE == "parallel_pfc":
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

    du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)


if __name__ == "__main__":
    main()
