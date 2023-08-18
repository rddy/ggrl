import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ExpDataset(Dataset):
    def __init__(self, trajs):
        self.trajs = []
        self.obses = []
        self.actions = []
        self.idxes_of_traj = []
        self.returns = []
        for traj in trajs:
            self.put(traj)

    def put(self, traj):
        traj_len = len(traj)
        obses, actions, rewards = list(zip(*traj))[:3]
        returns = np.cumsum(rewards[::-1])[::-1]
        max_return = returns.max()
        traj_idx = len(self.idxes_of_traj)
        n_prev_obses = len(self.obses)
        n_new_obses = len(obses)
        self.idxes_of_traj.append((n_prev_obses, n_prev_obses + n_new_obses))
        self.returns.extend(returns)
        self.actions.extend(actions)
        self.obses.extend(obses)
        self.trajs.append(traj)

    def __getitem__(self, idx):
        return self.obses[idx], self.actions[idx], self.returns[idx]

    def __len__(self):
        return len(self.obses)

    def split(self, train_frac=0.9):
        n_trajs = len(self.trajs)
        assert n_trajs > 1
        idxes = np.arange(n_trajs)
        np.random.shuffle(idxes)
        n_train = int(train_frac * n_trajs)
        n_train = np.clip(n_train, 1, n_trajs - 1)
        agg = lambda idxes: sum(
            [list(range(*self.idxes_of_traj[traj_idx])) for traj_idx in idxes], []
        )
        train_idxes = agg(idxes[:n_train])
        val_idxes = agg(idxes[n_train:])
        self.train = Subset(self, train_idxes)
        self.val = Subset(self, val_idxes)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.trajs, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            trajs = pickle.load(f)
        return cls(trajs)
