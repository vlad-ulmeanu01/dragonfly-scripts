import numpy as np
import random
import time
import json
import sys
import os
import re


INCAST_TYPE = "group" # "host"
assert INCAST_TYPE in ["host", "group"], "unknown INCAST_TYPE"


ROOT = "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/dragonfly-scripts" # "/home/vlad/Documents/SublimeMerge/dragonfly-scripts"
TM_FILE = os.path.join(ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/experiment_{INCAST_TYPE}_incast_tmp.cm")
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

# K = 6
# TOPOS = {
#     "iulian": [None for _ in range(5)],
#     "0": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_0.txt") for i in range(1, 6)],
#     "-20": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-20.txt") for i in range(1, 6)],
#     "-40": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-40.txt") for i in range(1, 6)],
#     "-60": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-60.txt") for i in range(1, 6)],
#     "-80": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-80.txt") for i in range(1, 6)]
# }

K = 8
TOPOS = {
    "iulian": [None for _ in range(5)],
    "-14": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-14.txt") for i in range(1, 6)],
    "-76": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-76.txt") for i in range(1, 6)],
    "-138": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-138.txt") for i in range(1, 6)],
    "-200": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-200.txt") for i in range(1, 6)],
    "-264": [os.path.join(CFG_ROOT, f"k_{K}", f"config_{i}_score_-264.txt") for i in range(1, 6)]
}

# K = 8
# TOPOS = {
#     "-14": os.path.join(CFG_ROOT, f"k_{K}", "config_1_score_-14.txt"),
#     "-258": os.path.join(CFG_ROOT, f"k_{K}", "config_4_score_-258.txt"),
#     "iulian": None
# }


FLOWSIZE = 2 * 10**6 # 2MB / host
CNT_RUNS_PER_TOPO = 5 # 30
BDP_PKTS = 33

QUEUE_SIZE = 3 * BDP_PKTS
ECN = (7, 26)

H = K // 2
CNT_NODES = (H**2 + 1) * H**2
CNT_GROUPS = H**2 + 1
GROUP_SIZE = H**2


def generate_transport_matrix(incasted_host: int):
    with open(TM_FILE, 'w') as fout:
        fout.write(f"Nodes {CNT_NODES}\n")
        fout.write(f"Connections {CNT_NODES - 1 if INCAST_TYPE == 'host' else CNT_NODES - GROUP_SIZE}\n")

        j = 1
        if INCAST_TYPE == "host":
            for i in range(CNT_NODES):
                if i != incasted_host:
                    fout.write(f"{i}->{incasted_host} id {j} start {0} size {FLOWSIZE}\n")
                    j += 1
        else: # group incast.
            incasted_host -= incasted_host % GROUP_SIZE
            for i in range(CNT_NODES):
                if i // GROUP_SIZE != incasted_host // GROUP_SIZE:
                    to = random.randrange(incasted_host, incasted_host + GROUP_SIZE)
                    fout.write(f"{i}->{to} id {j} start {0} size {FLOWSIZE}\n")
                    j += 1


class SimResult:
    def __init__(self, fcts: list, rtx: int):
        self.fcts = fcts
        self.rtx = rtx


def run_sim(topos: list, queue_size: int, ecn: tuple, end_time: int, cnt_paths: int, do_sender_cc: bool):
    srs = []

    t_start = time.time()
    for incasted_host in range(0, CNT_NODES, GROUP_SIZE):
        generate_transport_matrix(incasted_host)

        for topo in topos:
            cmd = ' '.join([
                EXE,
                f"-seed {random.randrange(1 << 30)}",
                f"-tm {TM_FILE}",
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

            print(f"Finished {topo = }, group {incasted_host // H**2} / {CNT_GROUPS}. {round(time.time() - t_start, 3)} s passed.", flush = True)

    return srs


def main():
    ht = {"INCAST_TYPE": INCAST_TYPE, "K": K, "CNT_RUNS_PER_TOPO": CNT_RUNS_PER_TOPO, "queue_size": QUEUE_SIZE, "ecn": list(ECN)}

    t_start = time.time()
    for topo_name in TOPOS:
        srs = run_sim(topos = TOPOS[topo_name], queue_size = QUEUE_SIZE, ecn = ECN, end_time = 2 * 10**5, cnt_paths = 128, do_sender_cc = True)
        
        # ht[topo_name] = {"mean_fcts": np.array([sr.fcts for sr in srs]).mean(axis = 0).tolist(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        ht[topo_name] = {"mean_fct": np.array([sr.fcts[-1] for sr in srs]).mean(), "mean_rtx": np.array([sr.rtx for sr in srs]).mean()}
        
        print(f"Finished {topo_name = }. {ht[topo_name] = }. {round(time.time() - t_start, 3)} s passed.", flush = True)

    with open(f"full_incast_{int(time.time())}.json", 'w') as fout:
        json.dump(ht, fout, indent = 4)


if __name__ == "__main__":
    main()
