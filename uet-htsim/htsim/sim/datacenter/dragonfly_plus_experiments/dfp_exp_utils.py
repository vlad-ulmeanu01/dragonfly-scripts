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

FLOW_SIZE = 2 * 10**6 # 2MB / host
LINK_SPEED = 10**5 # 100 Mbps
END_TIME = 2 * 10**5 # us
CNT_PATHS = 128
DO_SENDER_CC = True

BDP_PKTS = 33 # specific pentru Dfp. fa alt fisier utils pentru slimfly/dfly normal.

QUEUE_SIZE = BDP_PKTS # 3 * 
ECN = (int(BDP_PKTS * 0.2), BDP_PKTS - int(BDP_PKTS * 0.2))

PKT_SPRAYING = "greedy2"
assert PKT_SPRAYING in ["greedy1", "greedy2"], "unknown PKT_SPRAYING"


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
    do_sender_cc: bool, pkt_spraying: str, logout_fname: str
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
        "-q", f"{queue_size}",
        "-ecn", f"{ecn[0]}", f"{ecn[1]}",
        "-cwnd", f"37",
        "-topo_type", "DFP_SPARSE",
        "-sender_cc_only" if do_sender_cc else "-receiver_cc_only",
        "-load_balancing_algo", "oblivious",
        "-strat", "ecmp_all"
    ]

    if topo:
        cmdlist.extend(["-topo_dfp_sparse", f"{topo}"])
    if pkt_spraying == "greedy2": # sper ca baga Greedy[2] asta.
        cmdlist.extend(["-ar_method", "queue"])

    return cmdlist


def post_run_cleanup(cnt_runs_per_topo: int):
    os.system(f"rm {TM_FILE}")
    os.system("rm idmap.txt")

    for root, dir, files in os.walk('.'):
        for file in files:
            if file.startswith(f"logout_{RUN_ID}") and file.endswith(".dat"):
                os.system(f"rm {file}")
