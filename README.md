# ggrl

Learn to [upside-down reinforcement learn](https://arxiv.org/abs/1912.02875) using a [Neural Process](https://arxiv.org/abs/1807.01622).
Key idea: instead of conditioning the action predictor on the return, condition on a set of examples (state, action, return) with lower returns.

![](./cartpole.png)