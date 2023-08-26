import torch
import torch.nn as nn
import torch.nn.functional as F

import networks
import utils


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
        self.n_act_dims = n_act_dims
        self.discrete = discrete
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
        actions = utils.featurize_actions(actions, self.n_act_dims, self.discrete)
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


class PVN(Module):
    def __init__(
        self,
        n_obs_dims,
        n_act_dims,
        n_emb_dims,
        discrete,
        emb_layer_kwargs,
        pred_layer_kwargs,
        recon_layer_kwargs,
    ):
        super().__init__()
        self.n_act_dims = n_act_dims
        self.discrete = discrete
        self.emb_layer = networks.build_mlp(
            n_in_dims=n_obs_dims + n_act_dims, n_out_dims=n_emb_dims, **emb_layer_kwargs
        )
        self.pred_layer = networks.build_mlp(
            n_in_dims=n_emb_dims, n_out_dims=1, **pred_layer_kwargs
        )
        self.recon_layer = networks.build_mlp(
            n_in_dims=n_emb_dims + n_obs_dims,
            n_out_dims=n_act_dims,
            **recon_layer_kwargs
        )

    def forward(self, obses, actions):
        emb_inpts = torch.cat([obses, actions], dim=2)
        embs = self.emb_layer(emb_inpts)
        embs = torch.mean(embs, dim=1)
        preds = self.pred_layer(embs)
        embs = embs[:, None, :]
        inner_batch_size = obses.shape[1]
        embs = embs.expand(-1, inner_batch_size, -1)
        recon_inpts = torch.cat([embs, obses], dim=2)
        recons = self.recon_layer(recon_inpts)
        return preds, recons


class MLP(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mlp = networks.build_mlp(*args, **kwargs)

    def forward(self, x):
        return self.mlp(x)
