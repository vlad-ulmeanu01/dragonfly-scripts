import time
import os


exps = [
    "full_permutation",
    # "partial_all_to_all",
    # "full_all_to_all",
    # "full_incast",
    # "adversarial_all_to_all",
    # "subclique_samez_all_to_all"
    # "full_all_to_all_worsen_early_completions"
]

ht_exps = {
    "full_permutation": [
        # "--K 4",
        # "--K 4 --DO_CC receiver",
        "--K 4 --DO_CC no_cc_unlimited",
        "--K 6",
        # "--K 6 --DO_CC receiver",
        "--K 6 --DO_CC no_cc_unlimited",
        "--K 8 --CNT_RUNS_PER_TOPO 3",
        # "--K 8 --DO_CC receiver --CNT_RUNS_PER_TOPO 3",
        # "--K 8 --DO_CC no_cc_unlimited --CNT_RUNS_PER_TOPO 3"
    ],
    "partial_all_to_all": [
        "--K 6",
        "--K 8 --CNT_RUNS_PER_TOPO 1",
    ],
    "full_all_to_all": [
        "--K 4 --ALL_TO_ALL_TYPE parallel",
        "--K 4 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 2000000",
        "--K 6 --ALL_TO_ALL_TYPE parallel --CNT_RUNS_PER_TOPO 1",
        "--K 6 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 2000000",
        "--K 8 --ALL_TO_ALL_TYPE parallel --CNT_RUNS_PER_TOPO 1",
        "--K 8 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 2000000",
    ],
    "full_incast": [
        "--K 4 --INCAST_TYPE group",
        "--K 4 --INCAST_TYPE group --DO_CC no_cc_pfc",
        # "--K 4 --INCAST_TYPE group --DO_CC no_cc_unlimited",
        "--K 6 --INCAST_TYPE group",
        "--K 6 --INCAST_TYPE group --DO_CC no_cc_pfc",
        # "--K 6 --INCAST_TYPE group --DO_CC no_cc_unlimited",
        "--K 8 --INCAST_TYPE group --CNT_RUNS_PER_TOPO 1",
        "--K 8 --INCAST_TYPE group --DO_CC no_cc_pfc --CNT_RUNS_PER_TOPO 1",
        # "--K 8 --INCAST_TYPE group --DO_CC no_cc_unlimited --CNT_RUNS_PER_TOPO 1",
    ],
    "adversarial_all_to_all": [
        "--K 4",
        # "--K 6",
        # "--K 8 --CNT_RUNS_PER_TOPO 1"
    ],
    "subclique_samez_all_to_all": [
        "--K 4",
        # "--K 6",
        # "--K 8 --CNT_RUNS_PER_TOPO 1"
    ],
    "full_all_to_all_worsen_early_completions": [
        # "--K 4 --CNT_RUNS_PER_TOPO 1",
        # "--K 6 --CNT_RUNS_PER_TOPO 1 --TOPOLOGIES_PER_SCORE 3 --END_TIME 2000000",
        "--K 8 --CNT_RUNS_PER_TOPO 1 --TOPOLOGIES_PER_SCORE 3 --END_TIME 20000000" # TODO incearca maine incast.
    ]
}

for exp_name in exps:
    for exp_opts in ht_exps[exp_name]:
        t_start = time.time()
        print(f"Starting {exp_name} {exp_opts}.")
        os.system(f"python3 {exp_name}.py {exp_opts}")
        print(f"Finished {exp_name} {exp_opts} in {round(time.time() - t_start, 3)} s.")
