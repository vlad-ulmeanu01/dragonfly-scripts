import torch.nn.functional as F
import numpy as np
import itertools
import torch
import copy

import utils

class PolicyNetA(torch.nn.Module):
    def __init__(self, G: int, KS: int, cnt_iters: int):
        super(PolicyNetA, self).__init__()

        self.cnt_iters = cnt_iters
        self.G = G

        self.conv_z = torch.nn.ModuleList([torch.nn.Conv3d(in_channels=1, out_channels=G, kernel_size=(G, 1, 1)) for _ in range(cnt_iters)])
        self.conv_j = torch.nn.ModuleList([torch.nn.Conv3d(in_channels=1, out_channels=G, kernel_size=(1, 1, G)) for _ in range(cnt_iters)])
        self.conv_all = torch.nn.ModuleList([torch.nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(KS, KS, KS), padding="same") for _ in range(cnt_iters)])

    def forward(self, x: torch.tensor):
        for i in range(self.cnt_iters):
            x_unsq = x.unsqueeze(0).unsqueeze(0) # (bs = 1, in = 1, G, G, G)

            out_cond1 = F.relu(self.conv_z[i](x_unsq).view(self.G, self.G, self.G)) # (bs = 1, out = G, z = 1, i = G, j = G)
            out_cond2 = F.relu(self.conv_j[i](x_unsq).view(self.G, self.G, self.G)) # (bs = 1, out = G, z = G, i = G, j = 1)
            out_all = F.relu(self.conv_all[i](x_unsq)[0].mean(dim = 0)) # (bs = 1, out = O, z = G, i = G, j = G)

            x = (out_cond1 + out_cond2 + out_all) / 3

        return x


class PolicyNetB(torch.nn.Module):
    def __init__(self):
        super(PolicyNetB, self).__init__()

        G = utils.G

        self.forbidden_mask = np.zeros((G, G, G), dtype = np.bool_)
        for z in range(G):
            self.forbidden_mask[z, z, :] = True
            self.forbidden_mask[z, :, z] = True
            self.forbidden_mask[:, z, z] = True
        self.forbidden_mask = torch.from_numpy(self.forbidden_mask.flatten())

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(G ** 3, G),
            torch.nn.ReLU(),
            torch.nn.Linear(G, G ** 3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(G ** 3, G),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(G, G ** 3)
        )

    def cast_input(self, X: torch.tensor):
        G = utils.G
        cast_X = np.zeros((len(X), G, G, G), dtype = np.float32)
        X_np = X.detach().numpy()

        X_np_og = copy.deepcopy(X_np)
        og_score = utils.evaluate_state(X_np)

        for z, i, j in itertools.product(range(G), range(G), range(G)):
            if z != i and z != j and i != j:
                # pun cast_X[bs, z, i, j] == cu recompensa pe care am primi-o daca facem (z, i, j).

                X_np = utils.do_move(X_np, np.array([(z, i, j) for _ in range(len(X))]))

                score = utils.evaluate_state(X_np)
                cast_X[:, z, i, j] = score - og_score

                del X_np
                X_np = copy.deepcopy(X_np_og)

        return torch.from_numpy(cast_X.reshape(len(X), -1))


    def forward(self, X: torch.tensor):
        return F.softmax(
            torch.where(self.forbidden_mask, -utils.inf, self.layers(self.cast_input(X))),
            dim = -1
        )
