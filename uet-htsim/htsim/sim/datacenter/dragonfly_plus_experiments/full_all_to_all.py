# Generate connection matrices for ALL_TO_ALL_TYPE = "serial" with: python3 gen_serial_alltoall.py dfp_all_to_all_k_{K}.cm N N N 2000000 0 0
# Where N is the node count = H**2 * (H**2 + 1)

import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


ALL_TO_ALL_TYPE = "parallel"
# ALL_TO_ALL_TYPE = "serial"
assert ALL_TO_ALL_TYPE in ["parallel", "serial"], "unknown ALL_TO_ALL_TYPE"

K = 4

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = 5
CNT_TMS_PER_TYPE = {"serial": 5, "parallel": 1}

TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


def run_sim(topos: list, tm_file: str):
    pool = mp.Pool(processes = 4)
    srs = []

    t_start = time.time()
    for topo in topos:
        cmds = [
            du.get_htsim_cmdlist(
                seed = du.SEEDS[nt], tm_file = tm_file, end_time = du.END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = du.QUEUE_SIZE,
                ecn = du.ECN, topo = topo, do_sender_cc = du.DO_SENDER_CC, pkt_spraying = du.PKT_SPRAYING, logout_fname = f"logout_{du.RUN_ID}_{nt}.dat"
            )
            for nt in range(CNT_RUNS_PER_TOPO[ALL_TO_ALL_TYPE])
        ]

        srs.extend(pool.map(du.proc_run, cmds))

        print(f"Finished {topo = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main():
    ht = {
        "EXP_TYPE": "full_all_to_all", "ALL_TO_ALL_TYPE": ALL_TO_ALL_TYPE, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "TOPOLOGIES_PER_SCORE": du.TOPOLOGIES_PER_SCORE,
        "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE, "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS,
        "DO_SENDER_CC": du.DO_SENDER_CC, "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": du.QUEUE_SIZE, "ECN": list(du.ECN), "PKT_SPRAYING": du.PKT_SPRAYING,
        "TOTAL_TIME": 0
    }

    t_start = time.time()
    for topo_name in du.TOPOS[K]:
        srs = []
        for tm_id in range(1, CNT_TMS_PER_TYPE[ALL_TO_ALL_TYPE] + 1):
            tm_file = os.path.join(TM_FOLDER, f"dfp_all_to_all_k_{K}_no_{tm_id}.cm" if ALL_TO_ALL_TYPE == "serial" else f"parallel_all_to_all_k_{K}_2MB.cm")
            srs.extend(run_sim(topos = du.TOPOS[K][topo_name], tm_file = tm_file))

        ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        
        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)


if __name__ == "__main__":
    main()
