import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ExpDataset(Dataset):
    def __init__(self):
        self.data = []
        self.obses = []
        self.actions = []
        self.idxes_of_traj = []
        self.returns = []

    def put_traj(self, traj):
        traj_len = len(traj)
        obses, actions, rewards = list(zip(*traj))
        returns = np.cumsum(rewards[::-1])[::-1]
        max_return = returns.max()
        traj_idx = self.num_trajs
        n_prev_obses = self.num_exps
        n_new_obses = len(obses)
        idxes = (n_prev_obses, n_prev_obses + n_new_obses)
        self.idxes_of_traj.append(idxes)
        if n_prev_obses == 0:
            update = lambda old, new: np.array(new)
        else:
            update = lambda old, new: np.concatenate((old, new), axis=0)
        self.obses = update(self.obses, obses)
        self.actions = update(self.actions, actions)
        self.returns = update(self.returns, returns)

    def put(self, traj):
        self.put_traj(traj)
        self.data.append(traj)

    def __getitem__(self, idx):
        return self.obses[idx], self.actions[idx], self.returns[idx]

    @property
    def num_exps(self):
        return len(self.obses)

    @property
    def num_trajs(self):
        return len(self.idxes_of_traj)

    def __len__(self):
        return self.num_exps

    def agg(self, traj_idxes):
        idxes = []
        for traj_idx in traj_idxes:
            exp_idxes = self.idxes_of_traj[traj_idx]
            idxes.extend(range(*exp_idxes))
        return idxes

    def num_split_groups(self):
        return self.num_trajs

    def split(self, train_frac=0.9):
        n = self.num_split_groups()
        assert n > 1
        idxes = np.arange(n)
        np.random.shuffle(idxes)
        n_train = int(train_frac * n)
        n_train = np.clip(n_train, 1, n - 1)
        train_idxes = self.agg(idxes[:n_train])
        val_idxes = self.agg(idxes[n_train:])
        self.train = Subset(self, train_idxes)
        self.val = Subset(self, val_idxes)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        dataset = cls()
        for x in data:
            dataset.put(x)
        return dataset


class PolicyEvalDataset(ExpDataset):
    def __init__(self, *args, inner_batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_batch_size = inner_batch_size
        self.traj_idxes_of_policy = []

    def put(self, trajs):
        n_prev_trajs = self.num_trajs
        n_new_trajs = len(trajs)
        traj_idxes = (n_prev_trajs, n_prev_trajs + n_new_trajs)
        self.traj_idxes_of_policy.append(traj_idxes)
        for traj in trajs:
            self.put_traj(traj)
        self.data.append(trajs)

    def __getitem__(self, policy_idx):
        traj_idxes = self.traj_idxes_of_policy[policy_idx]
        rtn_idxes = []
        exp_idxes = []
        for traj_idx in range(*traj_idxes):
            exp_idxes_of_traj = self.idxes_of_traj[traj_idx]
            rtn_idxes.append(exp_idxes_of_traj[0])
            exp_idxes.extend(range(*exp_idxes_of_traj))
        returns = self.returns[rtn_idxes]
        rtn = np.mean(returns)
        idxes = np.random.choice(exp_idxes, self.inner_batch_size)
        return self.obses[idxes], self.actions[idxes], rtn

    def __len__(self):
        return self.num_policies

    def agg(self, policy_idxes):
        return policy_idxes

    def num_split_groups(self):
        return self.num_policies

    @property
    def num_policies(self):
        return len(self.traj_idxes_of_policy)
