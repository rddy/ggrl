import torch.nn as nn


def build_mlp(n_in_dims, n_out_dims, n_layers, layer_size):
    layers = []
    n_layer_in_dims = n_in_dims
    for _ in range(n_layers):
        layers.append(nn.Linear(n_layer_in_dims, layer_size))
        layers.append(nn.ReLU())
        n_layer_in_dims = layer_size
    layers.append(nn.Linear(n_layer_in_dims, n_out_dims))
    return nn.Sequential(*layers)
