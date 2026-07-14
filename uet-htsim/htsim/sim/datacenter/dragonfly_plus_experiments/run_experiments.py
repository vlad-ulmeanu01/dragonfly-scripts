import time
import os


exps = [
    "full_permutation",
    # "partial_all_to_all",
    "full_all_to_all",
    "full_incast",
    # "adversarial_all_to_all",
    # "subclique_samez_all_to_all",
    # "full_all_to_who_worsen_early_completions",
    # "z_vs_y_symmetry",
]

ht_exps = {
    "full_permutation": [
        # "--K 4 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10",
        # "--K 6 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10",
        # "--K 8 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10",
        # "--K 4 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC receiver",
        # "--K 6 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC receiver",
        # "--K 8 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC receiver",
        # "--K 4 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_pfc",
        # "--K 6 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_pfc",
        # "--K 8 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_pfc",
        # "--K 4 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_unlimited",
        # "--K 6 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_unlimited",
        # "--K 8 --HT_TYPE standard --CNT_RUNS_PER_TOPO 10 --DO_CC no_cc_unlimited",
        "--K 12 --HT_TYPE standard",
        "--K 16 --HT_TYPE standard",
        "--K 20 --HT_TYPE standard",
    ],
    "partial_all_to_all": [
        "--K 6",
        "--K 8 --CNT_RUNS_PER_TOPO 1",
    ],
    "full_all_to_all": [
        # "--K 4 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_unlimited",
        # "--K 6 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_unlimited --END_TIME 2000000",
        # "--K 8 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_unlimited --END_TIME 20000000",
        # "--K 4 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 2000000",
        # "--K 6 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 2000000",
        # "--K 8 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --END_TIME 20000000",
        # "--K 4 --ALL_TO_ALL_TYPE parallel",
        # "--K 6 --ALL_TO_ALL_TYPE parallel --CNT_RUNS_PER_TOPO 1",
        # "--K 8 --ALL_TO_ALL_TYPE parallel --CNT_RUNS_PER_TOPO 1",
        # "--K 6 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --CNT_RUNS_PER_TOPO 1 --END_TIME 2000000",
        # "--K 8 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_pfc --CNT_RUNS_PER_TOPO 1 --END_TIME 20000000",
        # "--K 8 --ALL_TO_ALL_TYPE parallel_pfc --DO_CC no_cc_unlimited --CNT_RUNS_PER_TOPO 1 --END_TIME 20000000",
        "--K 12 --ALL_TO_ALL_TYPE parallel",
        "--K 16 --ALL_TO_ALL_TYPE parallel",
        "--K 20 --ALL_TO_ALL_TYPE parallel",
    ],
    "full_incast": [
        # "--K 4 --HT_FCT_KEEP all --INCAST_TYPE group",
        # "--K 6 --HT_FCT_KEEP all --INCAST_TYPE group",
        # "--K 8 --HT_FCT_KEEP all --INCAST_TYPE group",
        # "--K 8 --HT_FCT_KEEP all --INCAST_TYPE group --CNT_RUNS_PER_TOPO 1",
        
        # "--K 4 --HT_FCT_KEEP all --INCAST_TYPE group --DO_CC no_cc_pfc",
        # "--K 6 --HT_FCT_KEEP all --INCAST_TYPE group --DO_CC no_cc_pfc",
        # "--K 8 --HT_FCT_KEEP all --INCAST_TYPE group --DO_CC no_cc_pfc",

        # "--K 4 --INCAST_TYPE group",
        # # "--K 4 --INCAST_TYPE group --DO_CC no_cc_unlimited",
        # "--K 6 --INCAST_TYPE group",
        # "--K 6 --INCAST_TYPE group --DO_CC no_cc_pfc",
        # # "--K 6 --INCAST_TYPE group --DO_CC no_cc_unlimited",
        # "--K 8 --INCAST_TYPE group --CNT_RUNS_PER_TOPO 1",
        # "--K 8 --INCAST_TYPE group --DO_CC no_cc_pfc --CNT_RUNS_PER_TOPO 1",
        # # "--K 8 --INCAST_TYPE group --DO_CC no_cc_unlimited --CNT_RUNS_PER_TOPO 1",

        "--K 12 --HT_FCT_KEEP all --INCAST_TYPE group",
        "--K 16 --HT_FCT_KEEP all --INCAST_TYPE group",
        "--K 20 --HT_FCT_KEEP all --INCAST_TYPE group",
    ],
    "adversarial_all_to_all": [
        "--K 4",
        # "--K 6",
        # "--K 8 --CNT_RUNS_PER_TOPO 1",
    ],
    "subclique_samez_all_to_all": [
        "--K 4",
        # "--K 6",
        # "--K 8 --CNT_RUNS_PER_TOPO 1",
    ],
    "full_all_to_who_worsen_early_completions": [
        # "--K 4 --CNT_RUNS_PER_TOPO 1",
        # "--K 6 --CNT_RUNS_PER_TOPO 1 --TOPOLOGIES_PER_SCORE 3 --END_TIME 2000000",
        # "--K 8 --CNT_RUNS_PER_TOPO 1 --TOPOLOGIES_PER_SCORE 3 --END_TIME 20000000",
        "--K 4 --ALL_TO_WHO one",
        "--K 6 --ALL_TO_WHO one",
        "--K 8 --ALL_TO_WHO one",
    ],
    "z_vs_y_symmetry": [
        "--K 4 "
        "--TM_FILE /home/vlad/Documents/SublimeMerge/dragonfly-scripts/uet-htsim/htsim/sim/datacenter/dragonfly_plus_connection_matrices/SPARSE/wtf_adversarial_1.cm "
        "--BEST_Z_TOPO /home/vlad/Desktop/Probleme/LaburiDC/dragonfly_scripts/simulate_dfly_queue_sizes/configs/k_4/config_1_score_0.txt "
        "--WORST_Z_TOPO /home/vlad/Desktop/Probleme/LaburiDC/dragonfly_scripts/simulate_dfly_queue_sizes/configs/k_4/config_1_score_-10.txt"
    ]
}

for exp_name in exps:
    for exp_opts in ht_exps[exp_name]:
        t_start = time.time()
        print(f"Starting {exp_name} {exp_opts}.", flush = True)
        os.system(f"python3 {exp_name}.py {exp_opts}")
        print(f"Finished {exp_name} {exp_opts} in {round(time.time() - t_start, 3)} s.", flush = True)
