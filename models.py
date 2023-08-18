import torch
import torch.nn as nn
import torch.nn.functional as F

import networks


class Module(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def reset(self):
        def _reset(m):
            for c in m.children():
                if hasattr(c, "reset_parameters"):
                    c.reset_parameters()

        self.apply(_reset)


class UDRLNeuralProcess(Module):
    def __init__(
        self,
        n_obs_dims,
        n_act_dims,
        n_emb_dims,
        discrete,
        emb_layer_kwargs,
        pred_layer_kwargs,
    ):
        super().__init__()
        self.discrete = discrete
        self.n_act_dims = n_act_dims
        self.emb_layer = networks.build_mlp(
            n_in_dims=n_obs_dims + 1 + n_act_dims,
            n_out_dims=n_emb_dims,
            **emb_layer_kwargs
        )
        self.pred_layer = networks.build_mlp(
            n_in_dims=n_emb_dims + n_obs_dims,
            n_out_dims=n_act_dims,
            **pred_layer_kwargs
        )

    def _embed(self, obses, returns, actions):
        if self.discrete:
            actions = F.one_hot(actions.long(), num_classes=self.n_act_dims)
            actions = actions.type(torch.float32)
        cat = torch.cat([obses, returns[:, None], actions], dim=1)
        return self.emb_layer(cat)

    def embed(self, *args):
        embs = self._embed(*args)
        return torch.mean(embs, dim=0)

    def batch_embed(self, *args):
        embs = self._embed(*args)
        n_embs = embs.shape[0]
        pembs = torch.cumsum(embs, dim=0)
        denoms = torch.arange(1, n_embs + 1)[:, None]
        return pembs / denoms

    def forward(self, embs, obses):
        cat = torch.cat([embs, obses], dim=1)
        return self.pred_layer(cat)
