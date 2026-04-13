import multiprocessing as mp
import numpy as np
import time
import json
import os

import dfp_exp_utils as du


K = 8

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = 1

TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


def generate_parallel_tm(use_groups: list, tm_file: str):
    cnt_nodes = len(use_groups) * GROUP_SIZE
    use_nodes = [i for g in use_groups for i in range(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)]

    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {CNT_NODES}\n")
        fout.write(f"Connections {cnt_nodes * (cnt_nodes - 1)}\n")

        flow_id = 1
        for i in use_nodes:
            for j in use_nodes:
                if j != i:
                    fout.write(f"{i}->{j} id {flow_id} start {0} size {du.FLOW_SIZE}\n")
                    flow_id += 1


def run_sim(topos: list, bibd_file: str):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    tm_files = []
    with open(bibd_file) as fin:
        for i, line in enumerate(fin.readlines()):
            tm_files.append(os.path.join(TM_FOLDER, f"experiment_{du.RUN_ID}_{i}_tmp.cm"))
            generate_parallel_tm(use_groups = list(map(int, line.strip().split())), tm_file = tm_files[-1])

    t_start = time.time()
    for topo in topos:
        cmds = [
            du.get_htsim_cmdlist(
                seed = du.SEEDS[nt], tm_file = tm_file, end_time = du.END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = du.QUEUE_SIZE,
                ecn = du.ECN, topo = topo, do_cc = du.DO_CC, pkt_spraying = du.PKT_SPRAYING, logout_fname = f"logout_{du.RUN_ID}_{nt}_{id_tmf}.dat"
            )
            for nt in range(CNT_RUNS_PER_TOPO) for id_tmf, tm_file in enumerate(tm_files)
        ]

        srs.extend(pool.map(du.proc_run, cmds))

        print(f"Finished {topo = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    for tm_file in tm_files:
        os.system(f"rm {tm_file}")

    return srs


def main():
    for root, dir, files in os.walk(os.path.join(TM_FOLDER, "partial_all_to_all_BIBDs")):
        for file in files:
            if file.startswith("BIBD") and file.endswith(".txt") and int(file.split("n_")[1].split('_')[0]) == CNT_GROUPS:
                cnt_used_groups = int(file.split("k_")[1].split('_')[0])

                print(f"dbg {file = }, {K = }, {CNT_GROUPS = }, {cnt_used_groups = }")

                ht = {
                    "EXP_TYPE": "partial_all_to_all", "K": K, "CNT_USED_GROUPS": cnt_used_groups, "BIBD_file": file, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO,
                    "TOPOLOGIES_PER_SCORE": du.TOPOLOGIES_PER_SCORE, "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE,
                    "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS, "DO_CC": du.DO_CC,
                    "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": du.QUEUE_SIZE, "ECN": list(du.ECN), "PKT_SPRAYING": du.PKT_SPRAYING,
                    "TOTAL_TIME": 0
                }

                t_start = time.time()

                for topo_name in du.TOPOS[K]:
                    if topo_name in ["-138", "-200", "-264"]: # TODO del.
                        srs = run_sim(topos = du.TOPOS[K][topo_name], bibd_file = os.path.join(root, file))

                        ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
                        
                        print(f"Finished {topo_name = } with {cnt_used_groups = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

                ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

                with open(f"./jsons/partial_all_to_all_{int(time.time())}.json", 'w') as fout:
                    json.dump(ht, fout, indent = 4)

                du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)

                print(f"Finished {cnt_used_groups = }/{CNT_GROUPS}. {round(time.time() - t_start, 3)} s passed.", flush = True)


if __name__ == "__main__":
    main()
