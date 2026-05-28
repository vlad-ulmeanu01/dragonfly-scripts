from subprocess import Popen, PIPE
import multiprocessing as mp
import numpy as np
import itertools
import time
import json
import math
import os
import re

import dfp_exp_utils as du


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


def generate_group_incast_transport_matrix(args, incasted_host: int):
    with open(args.TM_FILE, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")
        fout.write(f"Connections {args.CNT_NODES - args.GROUP_SIZE}\n")

        j = 1
        incasted_host -= incasted_host % args.GROUP_SIZE
        to = incasted_host
        for i in range(args.CNT_NODES):
            if i // args.GROUP_SIZE != incasted_host // args.GROUP_SIZE:
                fout.write(f"{i}->{to} id {j} start {0} size {args.FLOW_SIZE}\n")
                to = to + 1 if to + 1 < incasted_host + args.GROUP_SIZE else incasted_host
                j += 1


def generate_worse_tm_file(args, sr: du.SimResult, rap_w: float, tm_file: str):
    with open(tm_file, 'w') as fout:
        fout.write(f"Nodes {args.CNT_NODES}\n")

        cnt_flows = args.CNT_NODES * (args.CNT_NODES - 1) if args.ALL_TO_WHO == "all" else args.CNT_NODES - args.GROUP_SIZE
        fout.write(f"Connections {cnt_flows}\n")

        # aici in special sr.fcts tine tuplu (fct, host_from, host_to).
        last_fct = sr.fcts[-1][0]
        fcts = sorted(sr.fcts[: math.ceil(int(len(sr.fcts) * rap_w))], key = lambda p: (-p[1], -p[2]))

        flows = []
        if args.ALL_TO_WHO == "one":
            incasted_host, to = args.INCASTED_HOST, args.INCASTED_HOST
            for i in range(args.CNT_NODES):
                if i // args.GROUP_SIZE != incasted_host // args.GROUP_SIZE:
                    flows.append((i, to))
                    to = to + 1 if to + 1 < incasted_host + args.GROUP_SIZE else incasted_host
        else:
            flows = [(i, j) for i in range(args.CNT_NODES) for j in range(args.CNT_NODES) if j != i]

        z = 1
        for i, j in flows:
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

    print(f"{topos[0] = }: finished srs_base. {round(time.time() - t_start, 3)} s passed.", flush = True)

    cmds = []
    for rap_w_id, rap_w in enumerate(args.WORSEN_RAPS):
        for (i, sr), ((tid, topo), nt) in zip(enumerate(srs_base), itertools.product(enumerate(topos), range(args.CNT_RUNS_PER_TOPO))):
            identif = f"{du.RUN_ID}_{rap_w_id}_{tid}_{nt}"
            if args.ALL_TO_WHO == "all":
                tm_file = os.path.join(args.TM_FOLDER, f"tmp_worse_tm_file_{identif}.cm")
                generate_worse_tm_file(args, sr, rap_w, tm_file)
                cmds.append(du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{identif}.dat"))
            else:
                for incasted_host in range(0, args.CNT_NODES, args.GROUP_SIZE):
                    tm_file = os.path.join(args.TM_FOLDER, f"tmp_worse_tm_file_{identif}_{incasted_host}.cm")
                    args.INCASTED_HOST = incasted_host
                    generate_worse_tm_file(args, sr, rap_w, tm_file)
                    cmds.append(du.get_htsim_cmdlist(args, args.SEEDS[nt], tm_file, topo, logout_fname = f"logout_{identif}_{incasted_host}.dat"))


    srs_worse = pool.map(du.proc_run, cmds)
    print(f"{topos[0] = }: finished srs_worse. {round(time.time() - t_start, 3)} s passed.", flush = True)

    rap_step = len(srs_worse) // len(args.WORSEN_RAPS)
    for rap_w, id_cmds_start in zip(args.WORSEN_RAPS, range(0, len(srs_worse), rap_step)):
        ht_srs[rap_w] = srs_worse[id_cmds_start: id_cmds_start + rap_step]

    ht_srs[0.0] = [du.SimResult([fct for fct, _, _ in sr.fcts], sr.rtx) for sr in srs_base]

    for rap_w_id, _ in enumerate(args.WORSEN_RAPS):
        for (tid, _), nt in itertools.product(enumerate(topos), range(args.CNT_RUNS_PER_TOPO)):
            identif = f"{du.RUN_ID}_{rap_w_id}_{tid}_{nt}"

            if args.ALL_TO_WHO == "all":
                tm_file = os.path.join(args.TM_FOLDER, f"tmp_worse_tm_file_{identif}.cm")
                os.system(f"rm {tm_file}")
            else:
                for incasted_host in range(0, args.CNT_NODES, args.GROUP_SIZE):
                    tm_file = os.path.join(args.TM_FOLDER, f"tmp_worse_tm_file_{identif}_{incasted_host}.cm")
                    os.system(f"rm {tm_file}")

    print(f"Finished {topos[0] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return ht_srs


def main(args):
    ht = du.get_default_ht(args)
    ht["EXP_TYPE"] = "full_all_to_who_worsen_early_competions"
    ht["ALL_TO_WHO"] = args.ALL_TO_WHO
    ht["WORSEN_RAPS"] = args.WORSEN_RAPS

    t_start = time.time()

    tm_file = os.path.join(args.TM_FOLDER, f"parallel_all_to_all_k_{args.K}_2MB.cm")
    if args.ALL_TO_WHO == "one":
        generate_group_incast_transport_matrix(args, incasted_host = 0)
        tm_file = args.TM_FILE

    for topo_name in args.TOPOS[args.K]:
        ht_srs = run_sim(
            args,
            topos = args.TOPOS[args.K][topo_name],
            tm_file = tm_file
        )

        for rap_w in ht_srs:
            cnt_flows = args.CNT_NODES * (args.CNT_NODES - 1) if args.ALL_TO_WHO == "all" else args.CNT_NODES - args.GROUP_SIZE

            srs = [sr for sr in ht_srs[rap_w] if len(sr.fcts) == cnt_flows]
            cnt_failed_sims = sum([len(sr.fcts) < cnt_flows for sr in ht_srs[rap_w]])

            ht[f"{topo_name}_{rap_w}"] = {
                "mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist() if len(srs) else [],
                "mean_rtx": np.array([sr.rtx for sr in srs]).mean() if len(srs) else [],
                "rap_failed_sims": cnt_failed_sims / len(ht_srs[rap_w])
            }

        print(f"Finished {topo_name = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    ht["TOTAL_TIME"] = round(time.time() - t_start, 3)

    with open(f"./jsons/full_all_to_who_worsen_early_competions_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)

    du.post_run_cleanup(args)


if __name__ == "__main__":
    parser = du.get_default_parser()

    parser.add_argument(
        "--ALL_TO_WHO", type = str, default = "all", choices = ["all", "one"],
        help = "(specific to full_all_to_who_worsen_early_competions.py) supports both all-to-all and all-to-one (incast)."
    )

    parser.add_argument(
        "--WORSEN_RAPS", type = str, default = "[0.05,0.1,0.15,0.2,0.3]",
        help = "(specific to full_all_to_who_worsen_early_competions.py) string array of floats (no spaces): the first ?% of finishing flows will be padded s.t. they finish at least at the actual FCT in a rerun."
    )

    args = parser.parse_args()
    args = du.edit_default_args(args)

    # other specific arg edits..
    args.TM_FOLDER = os.path.join(du.ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")
    args.WORSEN_RAPS = list(map(float, args.WORSEN_RAPS.split('[')[-1].split(']')[0].split(',')))

    main(args)
