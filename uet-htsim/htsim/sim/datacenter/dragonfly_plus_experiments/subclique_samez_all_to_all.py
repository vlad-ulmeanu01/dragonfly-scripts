import multiprocessing as mp
import numpy as np
import itertools
import random
import time
import json
import os

import dfp_exp_utils as du


K = 6

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = 5
MAX_SUBCLIQUES_PER_RUN_SIM = 10
SUBCLIQUE_SHUFFLE_SEED = 34893

TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


def generate_subclique_tm(clique_gs: list, tm_file: str):
    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {CNT_NODES}\n")
        fout.write(f"Connections {len(clique_gs) * (len(clique_gs) - 1) * GROUP_SIZE**2}\n")

        conn_id = 1
        for i, j in itertools.product(clique_gs, clique_gs):
            if i != j:
                for x, y in itertools.product(range(i * GROUP_SIZE, (i+1) * GROUP_SIZE), range(j * GROUP_SIZE, (j+1) * GROUP_SIZE)):
                    fout.write(f"{x}->{y} id {conn_id} start {0} size {du.FLOW_SIZE}\n")
                    conn_id += 1


def find_subcliques(topos: list):
    ht_tm = {} # ht_tm[z_same][len clique] = (topo = config file, tm_file).

    tm_id = 0
    seen_none = False
    for topo in topos:
        if topo is not None or not seen_none:
            Z = [[0 for j in range(CNT_GROUPS)] for i in range(CNT_GROUPS)]

            if topo is None:
                seen_none = True
                rows = [[[] for j in range(H)] for i in range(CNT_GROUPS)]
                for i in range(CNT_GROUPS):
                    for j in range(i+1, CNT_GROUPS):
                        src_sw, dst_sw = (j - 1) // H, i // H # src_sw, dst_sw are spine switch ids, local for each group.
                        rows[i][src_sw].append(j)
                        rows[j][dst_sw].append(i)
            else:
                rows = []
                with open(topo) as fin:
                    for line in fin.readlines():
                        row = list(map(int, line.strip().split()))
                        rows.append([row[i: i+H] for i in range(0, len(row), H)])

            for row in rows:
                for buck in row:
                    for i in range(H):
                        for j in range(i+1, H):
                            Z[min(buck[i], buck[j])][max(buck[i], buck[j])] += 1

            for mask in range(1 << CNT_GROUPS):
                clique_gs = [i for i in range(CNT_GROUPS) if mask & (1 << i)]
                zs = [Z[clique_gs[i]][clique_gs[j]] for i in range(len(clique_gs)) for j in range(i+1, len(clique_gs))]
                if zs and min(zs) == max(zs):
                    z_same = zs[0]
                    if z_same not in ht_tm:
                        ht_tm[z_same] = {}
                    if len(clique_gs) not in ht_tm[z_same]:
                        ht_tm[z_same][len(clique_gs)] = []
                    
                    tm_file = os.path.join(TM_FOLDER, f"experiment_{du.RUN_ID}_{tm_id}.cm")
                    generate_subclique_tm(clique_gs, tm_file)
                    ht_tm[z_same][len(clique_gs)].append((topo, tm_file))
                    tm_id += 1

    return ht_tm


def run_sim(arr_topos_tm_files: list):
    if len(arr_topos_tm_files) > MAX_SUBCLIQUES_PER_RUN_SIM:
        random.seed(SUBCLIQUE_SHUFFLE_SEED)
        random.shuffle(arr_topos_tm_files)
        arr_topos_tm_files = arr_topos_tm_files[: MAX_SUBCLIQUES_PER_RUN_SIM]

    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    cmds = [
        du.get_htsim_cmdlist(
            seed = du.SEEDS[nt], tm_file = tm_file, end_time = du.END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = du.QUEUE_SIZE,
            ecn = du.ECN, topo = topo, do_cc = du.DO_CC, pkt_spraying = du.PKT_SPRAYING, logout_fname = f"logout_{du.RUN_ID}_{nt}_{tid}.dat"
        )
        for nt in range(CNT_RUNS_PER_TOPO) for tid, (topo, tm_file) in enumerate(arr_topos_tm_files)
    ]

    srs = pool.map(du.proc_run, cmds)

    return srs


def main():
    ht = {
        "EXP_TYPE": "subclique_samez_all_to_all", "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "TOPOLOGIES_PER_SCORE": du.TOPOLOGIES_PER_SCORE,
        "MAX_SUBCLIQUES_PER_RUN_SIM": MAX_SUBCLIQUES_PER_RUN_SIM, "SUBCLIQUE_SHUFFLE_SEED": SUBCLIQUE_SHUFFLE_SEED,
        "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE, "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS,
        "DO_CC": du.DO_CC, "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": du.QUEUE_SIZE, "ECN": list(du.ECN), "PKT_SPRAYING": du.PKT_SPRAYING,
        "TOTAL_TIME": 0
    }

    t_start = time.time()

    ht_tm = find_subcliques(topos = [topo for topo_name in du.TOPOS[K] for topo in du.TOPOS[K][topo_name]])

    for z_same in ht_tm:
        ht[f"z_same_{z_same}"] = {}
        for len_clique in ht_tm[z_same]:
            srs = run_sim(ht_tm[z_same][len_clique])
            ht[f"z_same_{z_same}"][f"len_clique_{len_clique}"] = {"fcts": [sr.fcts for sr in srs], "rtxs": [sr.rtx for sr in srs]}
            print(f"Finished {z_same = }, {len_clique = }: mean FCT = {np.mean(ht[f'z_same_{z_same}'][f'len_clique_{len_clique}']['fcts'])}. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/subclique_samez_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    for z_same in ht_tm:
        for len_clique in ht_tm[z_same]:
            for _, tm_file in ht_tm[z_same][len_clique]:
                os.system(f"rm {tm_file}")

    du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)


if __name__ == "__main__":
    main()
