import abc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Optimizer(object):
    def __init__(self, model, dataset, batch_size=32, device="cpu", opt_kwargs={}):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), **opt_kwargs)

    @abc.abstractmethod
    def compute_loss(self, batch):
        raise NotImplementedError

    def compute_val_loss(self):
        val_losses = []
        val_dataloader = DataLoader(self.dataset.val, self.batch_size)
        for batch in val_dataloader:
            batch = self.format_batch(batch)
            loss = self.compute_loss(batch)
            val_losses.append(loss.detach().numpy())
        return np.mean(val_losses)

    def format_batch(self, batch):
        return [x.to(self.device) for x in batch]

    def train(self, n_epochs, verbose=False):
        self.model.train()
        train_dataloader = DataLoader(self.dataset.train, self.batch_size)
        for i in range(n_epochs):
            train_losses = []
            for batch in train_dataloader:
                batch = self.format_batch(batch)
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.detach().numpy())
            val_loss = self.compute_val_loss()
            train_loss = np.mean(train_losses)
            if verbose:
                print(
                    "epoch: %d | val loss: %f | train loss: %f"
                    % (i, val_loss, train_loss)
                )
        self.model.eval()


class UDRLNeuralProcessOptimizer(Optimizer):
    def format_batch(self, batch):
        obses, actions, returns = super().format_batch(batch)
        obses = obses.type(torch.float32)
        actions = actions.long()
        returns = returns.type(torch.float32)
        return (obses, actions, returns)

    def compute_loss(self, batch):
        obses, actions, returns = batch
        returns, idxes = torch.sort(returns)
        obses = obses[idxes]
        actions = actions[idxes]
        embs = self.model.batch_embed(obses[:-1], returns[:-1], actions[:-1])
        preds = self.model(embs, obses[1:])
        return F.cross_entropy(preds, actions[1:])
