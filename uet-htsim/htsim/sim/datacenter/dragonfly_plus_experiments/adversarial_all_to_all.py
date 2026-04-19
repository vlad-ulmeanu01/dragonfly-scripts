import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import os

import dfp_exp_utils as du


K = 4

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

CNT_RUNS_PER_TOPO = 5
PERC_ADV_PAIRS = 0.5

BEST_TOPO_NAMES = {4: '0', 6: '0', 8: "-14"}

TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


def generate_adversarial_tm(topo: str, tm_file: str, perc_adv_pairs: float, seed: int):
    Z = [[0 for j in range(CNT_GROUPS)] for i in range(CNT_GROUPS)]

    if topo is None:
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

    cnt_group_pairs = int(perc_adv_pairs * (CNT_GROUPS - 1) * CNT_GROUPS / 2)
    group_fv = [0 for _ in range(CNT_GROUPS)]
    group_pairs = set()
    while len(group_pairs) < cnt_group_pairs:
        sel_pair = None
        for i in range(CNT_GROUPS):
            for j in range(i+1, CNT_GROUPS):
                if (i, j) not in group_pairs and (
                    sel_pair is None or\
                    group_fv[i] + group_fv[j] < group_fv[sel_pair[0]] + group_fv[sel_pair[1]] or\
                    (group_fv[i] + group_fv[j] == group_fv[sel_pair[0]] + group_fv[sel_pair[1]] and Z[i][j] > Z[sel_pair[0]][sel_pair[1]])
                ):
                    sel_pair = (i, j)
        group_pairs.add(sel_pair)
        group_fv[sel_pair[0]] += 1
        group_fv[sel_pair[1]] += 1

    print(f"(dbg generate_adversarial_tm) {topo = }, {Z = }, {group_pairs = }")

    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {CNT_NODES}\n")
        # fout.write(f"Connections {cnt_group_pairs * GROUP_SIZE**2 * 2}\n")
        fout.write(f"Connections {cnt_group_pairs * GROUP_SIZE**2}\n")

        conn_id = 1
        for i, j in group_pairs:
            for x, y in itertools.product(range(i * GROUP_SIZE, (i+1) * GROUP_SIZE), range(j * GROUP_SIZE, (j+1) * GROUP_SIZE)):
                fout.write(f"{x}->{y} id {conn_id} start {0} size {du.FLOW_SIZE}\n")
                # fout.write(f"{y}->{x} id {conn_id+1} start {0} size {du.FLOW_SIZE}\n")
                # conn_id += 2
                conn_id += 1


def run_sim(arr_topos_tm_files: list):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    
    cmds = [
        du.get_htsim_cmdlist(
            seed = du.SEEDS[nt], tm_file = tm_file, end_time = du.END_TIME, cnt_paths = du.CNT_PATHS, link_speed = du.LINK_SPEED, k = K, queue_size = du.QUEUE_SIZE,
            ecn = du.ECN, topo = topo, do_cc = du.DO_CC, pkt_spraying = du.PKT_SPRAYING, logout_fname = f"logout_{du.RUN_ID}_{nt}_{tid}.dat"
        )
        for tid, (topo, tm_file) in enumerate(arr_topos_tm_files) for nt in range(CNT_RUNS_PER_TOPO)
    ]

    srs = pool.map(du.proc_run, cmds)

    return srs


def main():
    ht = {
        "EXP_TYPE": "adversarial_all_to_all", "PERC_ADV_PAIRS": PERC_ADV_PAIRS, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "TOPOLOGIES_PER_SCORE": du.TOPOLOGIES_PER_SCORE,
        "SEEDS": du.SEEDS[:CNT_RUNS_PER_TOPO], "FLOW_SIZE": du.FLOW_SIZE, "LINK_SPEED": du.LINK_SPEED, "END_TIME": du.END_TIME, "CNT_PATHS": du.CNT_PATHS,
        "DO_CC": du.DO_CC, "BDP_PKTS": du.BDP_PKTS, "QUEUE_SIZE": du.QUEUE_SIZE, "ECN": list(du.ECN), "PKT_SPRAYING": du.PKT_SPRAYING,
        "TOTAL_TIME": 0
    }

    t_start = time.time()
    
    topo_best = du.TOPOS[K][BEST_TOPO_NAMES[K]][0]
    for topo_name in ["-10"]: # du.TOPOS[K]:
        if topo_name != BEST_TOPO_NAMES[K]:
            arr_adv = []
            for topo in du.TOPOS[K][topo_name]:
                tm_file = os.path.join(TM_FOLDER, f"experiment_{du.RUN_ID}_{len(arr_adv)}.cm")
                generate_adversarial_tm(topo, tm_file, PERC_ADV_PAIRS, len(arr_adv))
                arr_adv.append((topo, tm_file))

            srs_best = run_sim([(topo_best, tm_file) for _, tm_file in arr_adv])
            srs_oth = run_sim(arr_adv)

            ht[topo_name] = {
                "cmp_fcts": [(sr_best.fcts[-1], sr_oth.fcts[-1]) for sr_best, sr_oth in zip(srs_best, srs_oth)],
                "cmp_rtxs": [(sr_best.rtx, sr_oth.rtx) for sr_best, sr_oth in zip(srs_best, srs_oth)]
            }

            for i in range(len(arr_adv)):
                tm_file = os.path.join(TM_FOLDER, f"experiment_{du.RUN_ID}_{i}.cm")
                os.system(f"rm {tm_file}")

            print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/adversarial_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(cnt_runs_per_topo = CNT_RUNS_PER_TOPO)


if __name__ == "__main__":
    main()
