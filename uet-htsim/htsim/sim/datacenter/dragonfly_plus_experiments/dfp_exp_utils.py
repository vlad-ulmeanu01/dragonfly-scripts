from subprocess import Popen, PIPE
import time
import os
import re


RUN_ID = int(time.time())

NODENAME = os.uname().nodename
NODENAME = "grid.pub.ro" if NODENAME.endswith("grid.pub.ro") else NODENAME

ROOT = {
    "vlad-TM1701": "/home/vlad/Documents/SublimeMerge/dragonfly-scripts",
    "grid.pub.ro": "/export/home/acs/stud/v/vlad_adrian.ulmeanu/Probleme/dragonfly-scripts"
}[NODENAME]

CNT_PROCESSES = {
    "vlad-TM1701": 4,
    "grid.pub.ro": 16
}[NODENAME]

EXE = os.path.join(ROOT, "uet-htsim/htsim/sim/build/datacenter/htsim_uec")
CFG_ROOT = os.path.join(ROOT, "simulate_dfly_queue_sizes/configs/")


TOPOLOGIES_PER_SCORE = 5
SEEDS = [46005871, 514420321, 553169759, 604623024, 1041518730] # [random.randrange(1 << 30) for _ in range(TOPOLOGIES_PER_SCORE)]

# e.g. TOPOS[k = 8][score = "-14"] tine minte 5 config files cu topologii care produc acel scor.
TOPOS = {
    k: {
        score: [None] * TOPOLOGIES_PER_SCORE if score == "default"
               else [os.path.join(CFG_ROOT, f"k_{k}", f"config_{i}_score_{score}.txt") for i in range(1, TOPOLOGIES_PER_SCORE+1)]
        for score in scores
    }

    for k, scores in zip(
        [4, 6, 8],
        list(map(lambda u: ["default"] + list(map(str, u)), [
            [0, -4, -6, -8, -10],
            [0, -20, -40, -60, -80],
            [-14, -76, -138, -200, -264]
        ]))
    )
}

# fisier temporar folosibil de un experiment, specific pentru el.
TM_FILE = os.path.join(ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/experiment_{RUN_ID}_tmp.cm")

DFP_MAX_HOPS = 8
DFP_RTT = 15
PKT_SIZE = 4 * 10**3 # 4KB

FLOW_SIZE = 2 * 10**6 # 2MB / host
LINK_SPEED = 10**5 # 100 Mbps
END_TIME = 2 * 10**5 # us
CNT_PATHS = 128

DO_CC = "sender"
# DO_CC = "receiver"
# DO_CC = "no_cc_pfc"
# DO_CC = "no_cc_unlimited"
assert DO_CC in ["sender", "receiver", "no_cc_pfc", "no_cc_unlimited"]

BDP_PKTS = 33 # specific pentru Dfp. fa alt fisier utils pentru slimfly/dfly normal.

QUEUE_SIZE = BDP_PKTS
ECN = (int(BDP_PKTS * 0.2), BDP_PKTS - int(BDP_PKTS * 0.2))

PFC_OFF, PFC_ON = int(BDP_PKTS // 2), BDP_PKTS


class SimResult:
    def __init__(self, fcts: list, rtx: int):
        self.fcts = fcts
        self.rtx = rtx


def proc_run(cmdlist):
    fcts, rtx = [], None
    for line in Popen(cmdlist, shell = False, stdout = PIPE).stdout.readlines():
        line = line.decode().strip()
        if re.search(r"finished at (\d+)", line):
            fcts.append(float(line.split("finished at ")[1].split(' ')[0]))
        if re.search(r"Rtx:", line):
            rtx = int(line.split("Rtx: ")[1].split(' ')[0])

    return SimResult(fcts, rtx)


def get_htsim_cmdlist(
    seed: int, tm_file: str, end_time: int, cnt_paths: int, link_speed: int, k: int, queue_size: int, ecn: tuple, topo: str,
    do_cc: str, logout_fname: str
):
    cmdlist = [
        EXE,
        "-o", logout_fname,
        "-seed", f"{seed}",
        "-tm", f"{tm_file}",
        "-end", f"{end_time}",
        "-paths", f"{cnt_paths}",
        "-linkspeed", f"{link_speed}",
        "-radix", f"{k}",
        # "-cwnd", f"{BDP_PKTS}",
        "-topo_type", "DFP_SPARSE",
        "-load_balancing_algo", "oblivious"
    ]

    if topo:
        cmdlist.extend(["-topo_dfp_sparse", f"{topo}"])

    if do_cc in ["sender", "receiver", "no_cc_unlimited"]:
        if do_cc == "no_cc_unlimited":
            with open(tm_file) as fin:
                cnt_flows = len([i for i, line in enumerate(fin.readlines()) if i > 1 and len(line.strip()) > 0])
            queue_size = cnt_flows * FLOW_SIZE // PKT_SIZE + 1 # = cnt_packets
            ecn = (queue_size, queue_size)

        cmdlist.extend([
            "-q", f"{queue_size}",
            "-ecn", f"{ecn[0]}", f"{ecn[1]}",
            "-sender_cc_only" if do_cc == "sender" else ("-receiver_cc_only" if do_cc == "receiver" else "-no_cc"),
            "-strat", "ecmp_all",
            "-ar_method", "queue"
        ])
    else: # do_cc == "no_cc_pfc"
        queue_size = (k + 1) * BDP_PKTS
        cmdlist.extend([
            "-q", f"{queue_size}",
            "-ecn", f"{queue_size}", f"{queue_size}",
            "-no_cc",
            "-pfc", f"{PFC_OFF}", f"{PFC_ON}",
            "-rto", f"{DFP_RTT + queue_size * DFP_MAX_HOPS}",
            "-strat", "ecmp_ar",
            "-ar_method", "pqb"
        ])

    return cmdlist


def post_run_cleanup(cnt_runs_per_topo: int):
    os.system(f"rm logout_{RUN_ID}_*.dat")
    os.system("rm idmap.txt")
    os.system(f"rm {TM_FILE}")
