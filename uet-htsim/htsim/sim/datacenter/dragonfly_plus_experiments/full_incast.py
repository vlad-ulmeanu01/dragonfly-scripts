import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


INCAST_TYPE = "group"
# INCAST_TYPE = "host"
assert INCAST_TYPE in ["host", "group"], "unknown INCAST_TYPE"

K = 4

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = 5


def generate_transport_matrix(incasted_host: int):
    with open(du.TM_FILE, 'w') as fout:
        fout.write(f"Nodes {CNT_NODES}\n")
        fout.write(f"Connections {CNT_NODES - 1 if INCAST_TYPE == 'host' else CNT_NODES - GROUP_SIZE}\n")

        j = 1
        if INCAST_TYPE == "host":
            for i in range(CNT_NODES):
                if i != incasted_host:
                    fout.write(f"{i}->{incasted_host} id {j} start {0} size {du.FLOW_SIZE}\n")
                    j += 1
        else: # group incast.
            incasted_host -= incasted_host % GROUP_SIZE
            to = incasted_host
            for i in range(CNT_NODES):
                if i // GROUP_SIZE != incasted_host // GROUP_SIZE:
                    fout.write(f"{i}->{to} id {j} start {0} size {du.FLOW_SIZE}\n")
                    to = to + 1 if to + 1 < incasted_host + GROUP_SIZE else incasted_host
                    j += 1


def run_sim(topos: list):
    pool = mp.Pool(processes = 4)
    srs = []

    t_start = time.time()
    for incasted_host in range(0, CNT_NODES, GROUP_SIZE):
        generate_transport_matrix(incasted_host)

        for topo in topos:
            cmds = [
                du.get_htsim_cmdlist(
                    seed = du.SEEDS[nt], tm_file = du.TM_FILE, end_time = du.END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = du.QUEUE_SIZE,
                    ecn = du.ECN, topo = topo, do_sender_cc = du.DO_SENDER_CC, pkt_spraying = du.PKT_SPRAYING, logout_fname = f"logout_{du.RUN_ID}_{nt}.dat"
                )
                for nt in range(CNT_RUNS_PER_TOPO)
            ]

            srs.extend(pool.map(du.proc_run, cmds))

            print(f"Finished {topo = }, group {incasted_host // H**2} / {CNT_GROUPS}. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main():
    ht = {
        "EXP_TYPE": "full_incast", "INCAST_TYPE": INCAST_TYPE, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "TOPOLOGIES_PER_SCORE": du.TOPOLOGIES_PER_SCORE,
        "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE, "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS,
        "DO_SENDER_CC": du.DO_SENDER_CC, "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": du.QUEUE_SIZE, "ECN": list(du.ECN), "PKT_SPRAYING": du.PKT_SPRAYING,
        "TOTAL_TIME": 0
    }

    t_start = time.time()
    for topo_name in du.TOPOS[K]:
        srs = run_sim(topos = du.TOPOS[K][topo_name])
        
        ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        # ht[topo_name] = {"mean_fct": np.array([sr.fcts[-1] for sr in srs]).mean(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}

        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_incast_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)


if __name__ == "__main__":
    main()
