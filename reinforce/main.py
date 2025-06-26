import torch.nn.functional as F
import numpy as np
import random
import torch

import design
import utils


G, K, EXPECTED, inf = utils.G, utils.K, utils.EXPECTED, utils.inf
MAX_EPISODE_LEN, NUM_EPISODES = utils.MAX_EPISODE_LEN, utils.NUM_EPISODES
SEED = utils.SEED

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def generate_new_start():
    X = np.zeros((G, G, G), dtype = np.float32)

    group_ids = list(range(G))
    for z in range(G):
        random.shuffle(group_ids)
        for group in np.array([i for i in group_ids if i != z]).reshape(K // 2, K // 2):
            for i in range(K // 2):
                for j in range(i+1, K // 2):
                    m, M = min(group[i], group[j]), max(group[i], group[j])
                    X[z, m, M] = X[z, M, m] = 1.0

    return torch.from_numpy(X)


def evaluate_state(X: np.array):
    score = 0
    for i in range(G):
        for j in range(i+1, G):
            score -= abs(X[:, i, j].sum() - EXPECTED)

    for z in range(G):
        for i in range(G):
            if i != z:
                score -= abs(X[z, i].sum() - EXPECTED)

    return score


def do_move(X: np.array, z, i, j):
    i_buck_inds = np.arange(G)[X[z, i, :] == 1.0]
    j_buck_inds = np.arange(G)[X[z, :, j] == 1.0]

    X[z, i, i_buck_inds] = 0.0; X[z, j, j_buck_inds] = 0.0
    X[z, j, i_buck_inds] = 1.0; X[z, i, j_buck_inds] = 1.0

    return X


def rollout_episode(pnet, X: torch.tensor):
    log_probas, state_scores = [], []
    max_state_score, best_X = -inf, None

    for t in range(MAX_EPISODE_LEN):
        policy_proba_dist = pnet(X)
        
        state_scores.append(evaluate_state(X.detach().numpy()))

        if state_scores[-1] > max_state_score:
            max_state_score = state_scores[-1]
            best_X = X.detach()

        # sample urmatoarea actiune: schimb in grupul g pozitia grupurilor i si j.

        while True:
            with torch.no_grad():
                ind = F.softmax(policy_proba_dist, dim = 0).multinomial(num_samples = 1).item()
            z, i, j = ind // G**2, (ind % G**2) // G, ind % G
            if z != i and z != j and i != j and X[z, i, j] == 0.0:
                break
    
        log_probas.append(policy_proba_dist[ind])

        X_next = do_move(X.detach().clone().numpy(), z, i, j)
        X = torch.from_numpy(X_next)

    # adaug si state_scores pentru ultima actiune.
    state_scores.append(evaluate_state(X.detach().numpy()))

    gains = [state_scores[t+1] - state_scores[t] for t in range(MAX_EPISODE_LEN)]
    for t in range(len(gains) - 2, -1, -1):
        gains[t] += gains[t+1]

    log_probas = torch.tensor(gains) * torch.log(torch.stack(log_probas) + 1e-10)

    return log_probas, max_state_score, best_X


def main():
    pnet = design.PolicyNetB()
    optimizer = torch.optim.Adam(params = pnet.parameters())

    best_episode_score = -inf
    for cnt_episode in range(1, NUM_EPISODES + 1):
        X = generate_new_start()
        log_probas, max_state_score, best_X = rollout_episode(pnet, X)

        best_episode_score = max(best_episode_score, max_state_score)
        print(f"{cnt_episode = }, {best_episode_score = }", flush = True)

        cumul_lp = -log_probas.sum()

        optimizer.zero_grad()
        cumul_lp.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
