# Generate connection matrices with: python3 gen_serial_alltoall.py dfp_all_to_all_k_{K}.cm N N N 2000000 0 0
# Where N is the node count = H**2 * (H**2 + 1)

import numpy as np
import random
import time
import json
import sys
import os
import re


ALL_TO_ALL_TYPE = "parallel" # "serial"
assert ALL_TO_ALL_TYPE in ["parallel", "serial"], "unknown ALL_TO_ALL_TYPE"


ROOT = "/home/vlad/Documents/SublimeMerge/dragonfly-scripts"
# ROOT = "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/dragonfly-scripts"
OUTFILE = os.path.join(ROOT, "uet-htsim/htsim/sim/build/datacenter/test_out.txt")
EXE = os.path.join(ROOT, "uet-htsim/htsim/sim/build/datacenter/htsim_uec")
CFG_ROOT = os.path.join(ROOT, "simulate_dfly_queue_sizes/configs/")


# K = 4
# TOPOS = {
#     "iulian": [None for _ in range(5)],
#     "0": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_0.txt") for i in range(1, 6)],
#     "-4": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-4.txt") for i in range(1, 6)],
#     "-6": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-6.txt") for i in range(1, 6)],
#     "-8": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-8.txt") for i in range(1, 6)],
#     "-10": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-10.txt") for i in range(1, 6)]
# }

K = 6
TOPOS = {
    "iulian": [None for _ in range(5)],
    "0": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_0.txt") for i in range(1, 6)],
    "-20": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-20.txt") for i in range(1, 6)],
    "-40": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-40.txt") for i in range(1, 6)],
    "-60": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-60.txt") for i in range(1, 6)],
    "-80": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-80.txt") for i in range(1, 6)]
}

# K = 8
# TOPOS = {
#     "iulian": [None for _ in range(5)],
#     "-14": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-14.txt") for i in range(1, 6)],
#     "-76": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-76.txt") for i in range(1, 6)],
#     "-138": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-138.txt") for i in range(1, 6)],
#     "-200": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-200.txt") for i in range(1, 6)],
#     "-264": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-264.txt") for i in range(1, 6)]
# }


TM_FOLDER = os.path.join(ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/")


CNT_RUNS_PER_TOPO = 1 # 5
BDP_PKTS = 33

QUEUE_SIZE = 3 * BDP_PKTS
ECN = (int(BDP_PKTS * 0.2), int(BDP_PKTS * 0.8))

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2

PKT_SPRAYING = "greedy2"
assert PKT_SPRAYING in ["greedy1", "greedy2"], "unknown PKT_SPRAYING"


class SimResult:
    def __init__(self, fcts: list, rtx: int):
        self.fcts = fcts
        self.rtx = rtx


def run_sim(topos: list, tm_file: str, queue_size: int, ecn: tuple, end_time: int, cnt_paths: int, do_sender_cc: bool):
    srs = []

    t_start = time.time()
    for topo in topos:
        cmd = ' '.join([
            EXE,
            f"-seed {random.randrange(1 << 30)}",
            f"-tm {tm_file}",
            f"-end {end_time}",
            f"-paths {cnt_paths}",
            "-linkspeed 100000",
            f"-radix {K}",
            f"-q {queue_size}",
            f"-ecn {ecn[0]} {ecn[1]}",
            f"-cwnd 37",
            "-topo_type DFP_SPARSE",
            f"-topo_dfp_sparse {topo}" if topo else '',
            "-sender_cc_only" if do_sender_cc else "-receiver_cc_only",
            "-load_balancing_algo oblivious",
            "-strat ecmp_all",
            "-ar_method queue" if PKT_SPRAYING == "greedy2" else '', # sper ca baga Greedy[2] asta.
            f"> {OUTFILE}"
        ])

        for nt in range(CNT_RUNS_PER_TOPO):
            os.system(cmd)

            with open(OUTFILE) as fin:
                fcts, rtx = [], None
                for line in fin.readlines():
                    line = line.strip()
                    if re.search(r"finished at (\d+)", line):
                        fcts.append(float(line.split("finished at ")[1].split(' ')[0]))
                    if re.search(r"Rtx:", line):
                        rtx = int(line.split("Rtx: ")[1].split(' ')[0])
            
            srs.append(SimResult(fcts, rtx))

        print(f"Finished {topo = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main():
    ht = {"ALL_TO_ALL_TYPE": ALL_TO_ALL_TYPE, "PKT_SPRAYING": PKT_SPRAYING, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "queue_size": QUEUE_SIZE, "ecn": list(ECN)}

    t_start = time.time()
    for topo_name in TOPOS:
        for tm_id in (range(1, 6) if ALL_TO_ALL_TYPE == "serial" else [1]):
            tm_file = os.path.join(TM_FOLDER, f"dfp_all_to_all_k_{K}_no_{tm_id}.cm" if ALL_TO_ALL_TYPE == "serial" else f"parallel_all_to_all_k_{K}_2MB.cm")
            srs = run_sim(topos = TOPOS[topo_name], tm_file = tm_file, queue_size = QUEUE_SIZE, ecn = ECN, end_time = 2 * 10**5, cnt_paths = 128, do_sender_cc = True)

        ht[topo_name] = {"mean_fct": np.array([sr.fcts[-1] for sr in srs]).mean(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        
        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    with open(f"all_to_all_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)


if __name__ == "__main__":
    main()
