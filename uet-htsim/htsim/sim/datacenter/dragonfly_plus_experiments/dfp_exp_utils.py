from subprocess import Popen, PIPE
import argparse
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


def get_default_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--K", type = int, default = 4, help = "Topology switch radix. Must be even and at least 4.")
    
    parser.add_argument("--CNT_RUNS_PER_TOPO", type = int, default = 5, help = "How many different runs do we want to do on the same topology (<= 5). Each uses a different fixed seed from SEEDS.")
    parser.add_argument("--TOPOLOGIES_PER_SCORE", type = int, default = 5, help = "How many different topologies with the same symmetry score do we want to sample.")

    parser.add_argument(
        "--TOPOS_BEST_RAND_DEF_ONLY", type = str, default = "True", choices = ["True", "False"],
        help = "Consider only 3 symmetry scores, e.g. Best, Random, Default? Or 5."
    )

    parser.add_argument(
        "--SEEDS", type = str,
        default = "[46005871,514420321,553169759,604623024,1041518730,865395303,939417978,115462574,860852863,655131368]",
        help="Seeds used for different runs on the same topology. Pass as string with no spaces, e.g. [1,3,5]."
    )

    parser.add_argument("--DFP_MAX_HOPS", type = int, default = 8, help = "Max hops that can be taken by a packet before reaching the destination.")
    parser.add_argument("--DFP_RTT", type = int, default = 15, help = "Max round trip time for a packet (us).") # us
    parser.add_argument("--PKT_SIZE", type = int, default = 4 * 10**3, help = "Packet size (bytes).") # 4KB
    parser.add_argument("--FLOW_SIZE", type = int, default = 2 * 10**6, help = "Flow size (bytes).") # 2MB / host
    parser.add_argument("--LINK_SPEED", type = int, default = 10**5, help = "Link speed (bits per second)") # 100 Mbps
    parser.add_argument("--END_TIME", type = int, default = 2 * 10**5, help = "Simulation end time (us).")
    parser.add_argument("--CNT_PATHS", type = int, default = 128, help = "Max # paths a flow can be split into.")
    
    parser.add_argument(
        "--DO_CC", type = str, default = "sender",
        choices = ["sender", "receiver", "no_cc_pfc", "no_cc_unlimited"],
        help = "Congestion control variants: sender-only (NSCC), receiver-only, none (using PFC), none (queue sizes set to inf)."
    )

    parser.add_argument("--BDP_PKTS", type = int, default = 33, help = "# packets equivalent to the Bandwidth-Delay Product.")
    parser.add_argument("--QUEUE_SIZE", type = int, default = 33, help = "Switch max queue size. Default: 1 * BDP. Overridden by DO_CC = no_cc_pfc or no_cc_unlimited.")

    parser.add_argument("--ECN_RAP", type = float, default = 0.2, help = "Min queue size fraction to have nonzero probability to ECN tag a packet. (default: 0.2)")
    parser.add_argument("--ECN_LOW", type = int, default = int(33 * 0.2), help = "Min queue size to have nonzero probability to ECN tag a packet. (default: 20% BDP)")
    parser.add_argument("--ECN_HI", type = int, default = 33 - int(33 * 0.2), help = "Min queue size to have probability = 1 to ECN tag a packet. (default: 80% BDP)")

    parser.add_argument("--PFC_OFF", type = int, default = int(33 // 2), help = "Max queue size needed to stop PFC on a switch. (default: 50% BDP)")
    parser.add_argument("--PFC_ON", type = int, default = 33, help = "Min queue size needed to trigger PFC on a switch. (default: BDP)")

    parser.add_argument(
        "--TM_FILE",
        type = str,
        default = os.path.join(ROOT, f"uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/experiment_{RUN_ID}_tmp.cm"),
        help = "Temporary .cm file used by an experiment, specific for it."
    )

    return parser


def edit_default_args(args):
    args.SEEDS = list(map(int, args.SEEDS.split('[')[-1].split(']')[0].split(',')))

    args.H = args.K // 2
    args.CNT_NODES = (args.H**2 + 1) * args.H**2
    args.CNT_GROUPS = args.H**2 + 1
    args.GROUP_SIZE = args.H**2

    args.TOPOS_BEST_RAND_DEF_ONLY = (args.TOPOS_BEST_RAND_DEF_ONLY == "True")

    # e.g. TOPOS[k = 8][score = "-14"] remembers 5 config files with topologies having that score.
    args.TOPOS = {
        k: {
            score: [None] * args.TOPOLOGIES_PER_SCORE if score == "default"
                   else [os.path.join(CFG_ROOT, f"k_{k}", f"config_{i}_score_{score}.txt") for i in range(1, args.TOPOLOGIES_PER_SCORE+1)]
            for score in scores
        }

        for k, scores in zip(
            [4, 6, 8],
            list(map(lambda u: ["default"] + list(map(str, u)), [
                [0, -6] if args.TOPOS_BEST_RAND_DEF_ONLY else [0, -4, -6, -8, -10],
                [0, -40] if args.TOPOS_BEST_RAND_DEF_ONLY else [0, -20, -40, -60, -80],
                [-14, -138] if args.TOPOS_BEST_RAND_DEF_ONLY else [-14, -76, -138, -200, -264]
            ]))
        )
    }

    return args


def get_default_ht(args):
    ht = {
        "K": args.K,
        "CNT_RUNS_PER_TOPO": args.CNT_RUNS_PER_TOPO, "TOPOLOGIES_PER_SCORE": args.TOPOLOGIES_PER_SCORE,
        "SEEDS": args.SEEDS[:args.CNT_RUNS_PER_TOPO], "FLOW_SIZE": args.FLOW_SIZE, "LINK_SPEED": args.LINK_SPEED,
        "END_TIME": args.END_TIME, "CNT_PATHS": args.CNT_PATHS, "DO_CC": args.DO_CC,
        "BDP_PKTS": args.BDP_PKTS, "QUEUE_SIZE": args.QUEUE_SIZE, "ECN": [args.ECN_LOW, args.ECN_HI],
        "TOTAL_TIME": 0
    }

    if args.DO_CC == "no_cc_pfc":
        ht["QUEUE_SIZE"] = (args.K + 1) * args.BDP_PKTS
        ht["ECN"] = [int(args.ECN_RAP * ht["QUEUE_SIZE"]), int((1 - args.ECN_RAP) * ht["QUEUE_SIZE"])]
        ht["PFC"] = [args.PFC_OFF, args.PFC_ON]
        ht["RTO"] = args.DFP_RTT + ht["QUEUE_SIZE"] * args.DFP_MAX_HOPS
        ht["CWND"] = "inf"

    return ht


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


def get_htsim_cmdlist(args, seed: int, tm_file: str, topo: str, logout_fname: str):
    cmdlist = [
        EXE,
        "-o", logout_fname,
        "-seed", f"{seed}",
        "-tm", f"{tm_file}",
        "-end", f"{args.END_TIME}",
        "-paths", f"{args.CNT_PATHS}",
        "-linkspeed", f"{args.LINK_SPEED}",
        "-radix", f"{args.K}",
        "-topo_type", "DFP_SPARSE",
        "-load_balancing_algo", "oblivious"
    ]
    # "-cwnd", f"{BDP_PKTS}",

    if topo:
        cmdlist.extend(["-topo_dfp_sparse", f"{topo}"])

    if args.DO_CC in ["no_cc_unlimited", "no_cc_pfc"]:
        with open(tm_file) as fin:
            cnt_flows = len([i for i, line in enumerate(fin.readlines()) if i > 1 and len(line.strip()) > 0])
        inf_queue_size = cnt_flows * (args.FLOW_SIZE // args.PKT_SIZE + 1) # = cnt_packets

    if args.DO_CC in ["sender", "receiver", "no_cc_unlimited"]:
        queue_size = args.QUEUE_SIZE
        ecn = (args.ECN_LOW, args.ECN_HI)

        if args.DO_CC == "no_cc_unlimited":
            queue_size = inf_queue_size
            ecn = (queue_size, queue_size)
            cmdlist.extend(["-cwnd", f"{queue_size}"])

        cmdlist.extend([
            "-q", f"{queue_size}",
            "-ecn", f"{ecn[0]}", f"{ecn[1]}",
            "-sender_cc_only" if args.DO_CC == "sender" else ("-receiver_cc_only" if args.DO_CC == "receiver" else "-no_cc"),
            "-strat", "ecmp_all",
            # "-ar_method", "queue",
        ])
    else: # do_cc == "no_cc_pfc"
        queue_size = (args.K + 1) * args.BDP_PKTS
        cmdlist.extend([
            "-q", f"{queue_size}",
            "-ecn", f"{int(args.ECN_RAP * queue_size)}", f"{int((1 - args.ECN_RAP) * queue_size)}",
            # "-ecn", f"{queue_size}", f"{queue_size}",
            "-no_cc",
            "-pfc", f"{args.PFC_OFF}", f"{args.PFC_ON}",
            "-rto", f"{args.DFP_RTT + queue_size * args.DFP_MAX_HOPS}",
            "-strat", "ecmp_ar",
            # "-ar_method", "pqb",
            "-cwnd", f"{inf_queue_size}"
        ])

    print(f"{cmdlist = }")

    return cmdlist


def post_run_cleanup(args):
    os.system(f"rm logout_{RUN_ID}_*.dat")
    os.system("rm idmap.txt")
    os.system(f"rm {args.TM_FILE}")
