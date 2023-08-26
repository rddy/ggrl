import abc
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils


class Optimizer(object):
    def __init__(
        self, model, dataset, coeffs={}, batch_size=32, device="cpu", opt_kwargs={}
    ):
        self.model = model
        self.dataset = dataset
        self.coeffs = coeffs
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), **opt_kwargs)

    @abc.abstractmethod
    def compute_loss(self, batch):
        raise NotImplementedError

    def compute_val_loss(self):
        val_losses = defaultdict(list)
        val_dataloader = DataLoader(self.dataset.val, self.batch_size)
        for batch in val_dataloader:
            batch = self.format_batch(batch)
            loss = self.compute_loss(batch)
            for k, v in loss.items():
                val_losses[k].append(v.detach().numpy())
        return {k: np.mean(v) for k, v in val_losses.items()}

    def format_batch(self, batch):
        return [x.to(self.device) for x in batch]

    def train(self, n_epochs, verbose=False):
        self.model.train()
        train_dataloader = DataLoader(self.dataset.train, self.batch_size)
        for i in range(n_epochs):
            train_losses = defaultdict(list)
            for batch in train_dataloader:
                batch = self.format_batch(batch)
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                scalar_loss = torch.zeros(1)
                for k, v in loss.items():
                    coeff = self.coeffs.get(k, 1)
                    scalar_loss += coeff * v
                scalar_loss.backward()
                self.optimizer.step()
                for k, v in loss.items():
                    train_losses[k].append(v.detach().numpy())
            val_loss = self.compute_val_loss()
            train_loss = {k: np.mean(v) for k, v in train_losses.items()}
            if verbose:
                print("epoch: %d" % i)
                for k, v in train_loss.items():
                    print("train %s loss: %f" % (k, v))
                    print("val %s loss: %f" % (k, val_loss[k]))
                print("")
        self.model.eval()


class ExpOptimizer(Optimizer):
    def format_batch(self, batch):
        obses, actions, returns = super().format_batch(batch)
        obses = obses.type(torch.float32)
        actions = actions.long()
        returns = returns.type(torch.float32)
        return (obses, actions, returns)


class UDRLNeuralProcessOptimizer(ExpOptimizer):
    def compute_loss(self, batch):
        obses, actions, returns = batch
        returns, idxes = torch.sort(returns)
        obses = obses[idxes]
        actions = actions[idxes]
        embs = self.model.batch_embed(obses[:-1], returns[:-1], actions[:-1])
        preds = self.model(embs, obses[1:])
        targets = actions[1:]
        if self.model.discrete:
            pred_loss = F.cross_entropy(preds, targets)
        else:
            pred_loss = torch.mean((preds - targets) ** 2)
        loss = {"pred": pred_loss}
        return loss


class PVNOptimizer(ExpOptimizer):
    def __init__(self, *args, min_return=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_return = min_return

    @property
    def discrete(self):
        return self.model.discrete

    def compute_recon_loss(self, recons, actions):
        if self.discrete:
            n_act_dims = recons.shape[-1]
            recons = recons.view(-1, n_act_dims)
            return F.cross_entropy(recons, actions)
        else:
            return torch.mean((recons - actions) ** 2)

    def compute_loss(self, batch):
        obses, actions, returns = batch
        act_feats = utils.featurize_actions(
            actions, self.model.n_act_dims, self.model.discrete
        )
        preds, recons = self.model(obses, act_feats)
        shuffled_idxes = torch.randperm(actions.shape[1])
        contrast_preds, _ = self.model(obses, act_feats[:, shuffled_idxes])
        if self.discrete:
            actions = actions.ravel()
        loss = {}
        loss["recon"] = self.compute_recon_loss(recons, actions)
        loss["pred"] = torch.mean((preds - returns) ** 2)
        loss["contrast"] = torch.mean((contrast_preds - self.min_return) ** 2)
        return loss


class PVNPolicyOptimizer(PVNOptimizer):
    def __init__(self, *args, pvn, **kwargs):
        super().__init__(*args, **kwargs)
        self.pvn = pvn

    @property
    def discrete(self):
        return self.pvn.discrete

    def compute_loss(self, batch):
        obses = batch[0]
        n_obs_dims = obses.shape[-1]
        obses = obses.view(-1, n_obs_dims)
        actions = self.model(obses)
        if self.discrete:
            actions = F.softmax(actions, dim=-1)
        returns, recons = self.pvn(obses[None, :], actions[None, :])
        rtn = returns[0]
        recons = recons[0]
        loss = {}
        loss["recon"] = self.compute_recon_loss(recons, actions)
        loss["return"] = -rtn
        return loss
