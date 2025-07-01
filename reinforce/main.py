import torch.nn.functional as F
import numpy as np
import random
import torch
import numba
import time

import design
import utils


G, K, EXPECTED, BATCH_SIZE, inf = utils.G, utils.K, utils.EXPECTED, utils.BATCH_SIZE, utils.inf
MAX_EPISODE_LEN, NUM_EPISODES = utils.MAX_EPISODE_LEN, utils.NUM_EPISODES
SEED = utils.SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def generate_new_start(batch_size: int):
    X = np.zeros((batch_size, G, G, G), dtype = np.float32)

    for bid in range(batch_size):
        group_ids = list(range(G))
        for z in range(G):
            random.shuffle(group_ids)
            for group in np.array([i for i in group_ids if i != z]).reshape(K // 2, K // 2):
                for i in range(K // 2):
                    for j in range(i+1, K // 2):
                        m, M = min(group[i], group[j]), max(group[i], group[j])
                        X[bid, z, m, M] = X[bid, z, M, m] = 1.0

    return torch.from_numpy(X)


@numba.jit(nopython = True)
def evaluate_state(X: np.array):
    score = np.zeros(len(X), dtype = np.int32) # len(X) = batch_size.

    for i in range(G):
        for j in range(i+1, G):
            score -= np.abs(X[:, :, i, j].sum(axis = -1) - EXPECTED).astype(np.int32)

    for z in range(G):
        for i in range(G):
            if i != z:
                score -= np.abs(X[:, z, i].sum(axis = -1) - EXPECTED).astype(np.int32)

    return score


def do_move(X: np.array, actions: np.array):
    for bid, (z, i, j) in zip(range(len(X)), actions):
        if X[bid, z, i, j] == 0:
            i_buck_inds = np.arange(G)[X[bid, z, i, :] == 1.0]
            j_buck_inds = np.arange(G)[X[bid, z, :, j] == 1.0]

            X[bid, z, i, i_buck_inds] = 0.0; X[bid, z, i_buck_inds, i] = 0.0
            X[bid, z, j_buck_inds, j] = 0.0; X[bid, z, j, j_buck_inds] = 0.0

            X[bid, z, j, i_buck_inds] = 1.0; X[bid, z, i_buck_inds, j] = 1.0
            X[bid, z, j_buck_inds, i] = 1.0; X[bid, z, i, j_buck_inds] = 1.0

    return X


@numba.jit(nopython = True)
def compute_expert_actions(X: np.array, mask_expert_choices: np.array):
    expert_inds = np.zeros(len(X), dtype = np.int32) + (G + 2) # pot sa existe (pe langa starile unde nu vrem expert) stari care sunt maxim local. G + 2 = (0, 1, 2).

    for bid in range(len(X)):
        if mask_expert_choices[bid]:
            worst_diff, worst_ij = X[bid, :, 1, 2].sum() - EXPECTED, (1, 2)
            for i in range(G):
                for j in range(i+1, G):
                    diff = X[bid, :, i, j].sum() - EXPECTED
                    if abs(diff) > abs(worst_diff):
                        worst_diff, worst_ij = diff, (i, j)

            i, j = worst_ij
            if worst_diff < 0:
                z = np.random.choice(np.array([z for z in range(G) if X[bid, z, i, j] == 0]))
                expert_inds[bid] = z * G**2 + i * G + j
            elif worst_diff > 0:
                z = np.random.choice(np.array([z for z in range(G) if X[bid, z, i, j] == 1]))
                flip_i_with = np.random.choice(np.array([j for j in range(G) if X[bid, z, i, j] == 0 and j != i]))
                expert_inds[bid] = z * G**2 + i * G + flip_i_with

    return expert_inds


# proba_expert: 
def rollout_episode(pnet, X: torch.tensor, proba_expert: float):
    log_probas, state_scores = [], []
    max_state_score, best_X = -inf, None

    for t in range(MAX_EPISODE_LEN):
        policy_proba_dist = pnet(X)
        
        state_scores.append(evaluate_state(X.detach().numpy()))

        for bid in range(len(X)):
            if state_scores[-1][bid] > max_state_score:
                max_state_score = state_scores[-1][bid]
                best_X = X[bid].detach().clone()

        # sample urmatoarea actiune: schimb in grupul g pozitia grupurilor i si j.
        mask_expert_choices = (np.random.rand(len(X)) < proba_expert).astype(np.bool_)
        expert_inds = compute_expert_actions(X.detach().numpy(), mask_expert_choices)

        inds = np.where(
            mask_expert_choices,
            expert_inds,
            policy_proba_dist.multinomial(num_samples = 1).flatten().numpy()
        )
        arr_z, arr_i, arr_j = inds // G**2, (inds % G**2) // G, inds % G
    
        log_probas.append(policy_proba_dist[np.arange(len(X)), inds])

        X_next = do_move(X.detach().clone().numpy(), actions = np.vstack([arr_z, arr_i, arr_j]).T)
        X = torch.from_numpy(X_next)

    # adaug si state_scores pentru ultima actiune.
    state_scores.append(evaluate_state(X.detach().numpy()))

    gains = [state_scores[t+1] - state_scores[t] for t in range(MAX_EPISODE_LEN)]
    for t in range(len(gains) - 2, -1, -1):
        gains[t] += gains[t+1]

    gains = np.vstack(gains).T # (BS, EPISODE_LEN)
    gains = (gains - gains.mean(axis = 1).reshape(-1, 1)) / (gains.std(axis = 1).reshape(-1, 1) + 1e-10)

    log_probas = torch.tensor(gains) * torch.log(torch.vstack(log_probas).T + 1e-10) # (BS, EPISODE_LEN).

    return log_probas, max_state_score, best_X


def main():
    dbg_time_ht = {"rollout": 0.0, "backward": 0.0}

    pnet = design.PolicyNetB()
    optimizer = torch.optim.Adam(params = pnet.parameters())
    expert_sched = utils.DecayScheduler(start = 0.9, end = 0.1, decay = 500)

    best_episode_score, best_X = -inf, None
    for cnt_episode in range(1, NUM_EPISODES + 1):
        if best_episode_score == 0:
            break

        X = generate_new_start(batch_size = BATCH_SIZE)

        t_start = time.time()

        log_probas, ep_max_state_score, ep_best_X = rollout_episode(pnet, X, proba_expert = expert_sched.get(cnt_episode))
        
        dbg_time_ht["rollout"] += time.time() - t_start

        if best_episode_score < ep_max_state_score:
            best_episode_score = ep_max_state_score
            best_X = ep_best_X
        
        print(f"{cnt_episode = }, {best_episode_score = }, {ep_max_state_score = }", flush = True)

        t_start = time.time()

        cumul_lp = -log_probas.sum(dim = 1).mean()

        optimizer.zero_grad()
        cumul_lp.backward()
        optimizer.step()
        
        dbg_time_ht["backward"] += time.time() - t_start

        del log_probas, ep_max_state_score

        utils.print_debug_ht(dbg_time_ht)

    print(f"best_X = {utils.get_state_from_X(best_X)}")


if __name__ == "__main__":
    main()
