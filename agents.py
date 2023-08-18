import abc

import numpy as np
import torch


class Agent(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, obs):
        raise NotImplementedError

    def reward(self, reward):
        pass

    def reset(self):
        pass


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return self.env.action_space.sample()


class UDRLNeuralProcessAgent(Agent):
    def __init__(self, model, dataset, optimizer, train_freq=1, train_kwargs={}):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_freq = train_freq
        self.train_kwargs = train_kwargs
        self.traj = []
        self.training = False
        self.embed()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def embed(self):
        batch = [self.dataset.obses, self.dataset.actions, self.dataset.returns]
        batch = [torch.tensor(np.array(x)) for x in batch]
        obses, actions, returns = self.optimizer.format_batch(batch)
        emb = self.model.embed(obses, returns, actions)
        self.emb = emb[None, :]

    def act(self, obs):
        logits = self.model(self.emb, torch.tensor(obs)[None, :].type(torch.float32))
        distrn = torch.distributions.categorical.Categorical(logits=logits)
        action = distrn.sample()[0]
        action = action.detach().numpy()
        if self.training:
            self.last_obs = obs
            self.last_action = action
        return action

    def reward(self, reward):
        if self.training:
            assert self.last_obs is not None
            assert self.last_action is not None
            self.traj.append((self.last_obs, self.last_action, reward))
            self.last_obs = None
            self.last_action = None

    def reset(self):
        if self.training:
            if len(self.traj) > 0:
                self.dataset.put(self.traj)
                if len(self.dataset.trajs) % self.train_freq == 0:
                    self.model.reset()
                    self.dataset.split()
                    self.optimizer.train(**self.train_kwargs)
                    self.embed()
            self.traj = []
