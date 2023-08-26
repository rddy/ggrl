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


class LearningAgent(Agent):
    def __init__(self, model, dataset, optimizer, train_freq=1, train_kwargs={}):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.train_freq = train_freq
        self.train_kwargs = train_kwargs
        self.traj = []
        self.training = False

    def act(self, obs):
        action = self.eval_policy(obs)
        if self.model.discrete:
            distrn = torch.distributions.categorical.Categorical(logits=action)
            action = distrn.sample()
        action = action[0].detach().numpy()
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

    def train(self):
        self.model.reset()
        self.dataset.split()
        self.optimizer.train(**self.train_kwargs)

    def reset(self):
        self.traj = []


class UDRLNeuralProcessAgent(LearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed()

    def embed(self):
        batch = [self.dataset.obses, self.dataset.actions, self.dataset.returns]
        batch = [torch.tensor(np.array(x)) for x in batch]
        obses, actions, returns = self.optimizer.format_batch(batch)
        emb = self.model.embed(obses, returns, actions)
        self.emb = emb[None, :]

    def eval_policy(self, obs):
        obs = torch.tensor(obs)[None, :].type(torch.float32)
        return self.model(self.emb, obs)

    def reset(self):
        if self.training and len(self.traj) > 0:
            self.dataset.put(self.traj)
            if self.dataset.num_trajs % self.train_freq == 0:
                self.train()
                self.embed()
            self.traj = []


class PVNAgent(LearningAgent):
    def __init__(
        self, *args, policy, policy_optimizer, policy_train_kwargs={}, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.policy_train_kwargs = policy_train_kwargs
        self.trajs = []

    def train(self):
        super().train()
        self.policy.reset()
        self.policy_optimizer.train(**self.policy_train_kwargs)

    def eval_policy(self, obs):
        obs = torch.tensor(obs)[None, :].type(torch.float32)
        return self.policy(obs)

    def reset(self):
        if self.training and len(self.traj) > 0:
            self.trajs.append(self.traj)
            if len(self.trajs) == self.train_freq:
                self.dataset.put(self.trajs)
                self.train()
                self.trajs = []
            self.traj = []
