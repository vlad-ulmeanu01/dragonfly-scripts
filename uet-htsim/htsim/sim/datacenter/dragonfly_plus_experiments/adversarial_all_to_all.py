import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import os

import dfp_exp_utils as du


def generate_adversarial_tm(args, topo: str, tm_file: str, perc_adv_pairs: float, seed: int):
    Z = [[0 for j in range(args.CNT_GROUPS)] for i in range(args.CNT_GROUPS)]

    if topo is None:
        rows = [[[] for j in range(args.H)] for i in range(args.CNT_GROUPS)]
        for i in range(args.CNT_GROUPS):
            for j in range(i+1, args.CNT_GROUPS):
                src_sw, dst_sw = (j - 1) // args.H, i // args.H # src_sw, dst_sw are spine switch ids, local for each group.
                rows[i][src_sw].append(j)
                rows[j][dst_sw].append(i)
    else:
        rows = []
        with open(topo) as fin:
            for line in fin.readlines():
                row = list(map(int, line.strip().split()))
                rows.append([row[i: i+args.H] for i in range(0, len(row), args.H)])

    for row in rows:
        for buck in row:
            for i in range(args.H):
                for j in range(i+1, args.H):
                    Z[min(buck[i], buck[j])][max(buck[i], buck[j])] += 1

    cnt_group_pairs = int(perc_adv_pairs * (args.CNT_GROUPS - 1) * args.CNT_GROUPS / 2)
    group_fv = [0 for _ in range(args.CNT_GROUPS)]
    group_pairs = set()
    while len(group_pairs) < cnt_group_pairs:
        sel_pair = None
        for i in range(args.CNT_GROUPS):
            for j in range(i+1, args.CNT_GROUPS):
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
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {cnt_group_pairs * args.GROUP_SIZE**2}\n")

        conn_id = 1
        for i, j in group_pairs:
            for x, y in itertools.product(range(i * args.GROUP_SIZE, (i+1) * args.GROUP_SIZE), range(j * args.GROUP_SIZE, (j+1) * args.GROUP_SIZE)):
                fout.write(f"{x}->{y} id {conn_id} start {0} size {args.FLOW_SIZE}\n")
                conn_id += 1


def run_sim(args, arr_topos_tm_files: list):
    pool = mp.Pool(processes = du.CNT_PROCESSES)
    srs = []

    t_start = time.time()
    
    cmds = [
        du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{du.RUN_ID}_{tid}_{nt}.dat")
        for tid, (topo, tm_file) in enumerate(arr_topos_tm_files) for nt in range(args.CNT_RUNS_PER_TOPO)
    ]

    srs = pool.map(du.proc_run, cmds)

    return srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "adversarial_all_to_all"
    ht["PERC_ADV_PAIRS"] = args.PERC_ADV_PAIRS

    t_start = time.time()
    
    topo_best = args.TOPOS[args.K][args.BEST_TOPO_NAMES[args.K]][0]
    for topo_name in args.TOPOS[args.K]:
        if topo_name != args.BEST_TOPO_NAMES[args.K]:
            arr_adv = []
            for topo in args.TOPOS[args.K][topo_name]:
                tm_file = os.path.join(args.TM_FOLDER, f"experiment_{du.RUN_ID}_{len(arr_adv)}.cm")
                generate_adversarial_tm(args, topo, tm_file, args.PERC_ADV_PAIRS, len(arr_adv))
                arr_adv.append((topo, tm_file))

            srs_best = run_sim(args, [(topo_best, tm_file) for _, tm_file in arr_adv])
            srs_oth = run_sim(args, arr_adv)

            ht[topo_name] = {
                "cmp_fcts": [(sr_best.fcts[-1], sr_oth.fcts[-1]) for sr_best, sr_oth in zip(srs_best, srs_oth)],
                "cmp_rtxs": [(sr_best.rtx, sr_oth.rtx) for sr_best, sr_oth in zip(srs_best, srs_oth)]
            }

            for i in range(len(arr_adv)):
                tm_file = os.path.join(args.TM_FOLDER, f"experiment_{du.RUN_ID}_{i}.cm")
                os.system(f"rm {tm_file}")

            print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/adversarial_all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--PERC_ADV_PAIRS", type = float, default = 0.5,
        help = "(specific to adversarial_all_to_all.py, in [0.0, 1.0]). % of group pairs to select in the adversarial all-to-all traffic pattern. We select pairs with worse Z."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")
    args.BEST_TOPO_NAMES = {4: '0', 6: '0', 8: "-14"}

    main(args)
