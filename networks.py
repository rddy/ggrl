import torch.nn as nn


def mlp(n_in_dims, n_out_dims, n_layers, layer_size):
    layers = []
    n_layer_in_dims = n_in_dims
    for _ in range(n_layers):
        layers.append(nn.Linear(n_layer_in_dims, layer_size))
        layers.append(nn.ReLU())
        n_layer_in_dims = layer_size
    layers.append(nn.Linear(n_layer_in_dims, n_out_dims))
    return nn.Sequential(*layers)


class ConditionedMLP(nn.Module):
    def __init__(self, n_side_dims, n_in_dims, n_out_dims, n_layers, layer_size):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            mlp(n_in_dims, n_out_dims, n_layers, layer_size).children()
        )
        self.film_layers = nn.ModuleList(
            [
                nn.Linear(n_side_dims, layer_size * (layer_size + 1))
                for _ in range(n_layers)
            ]
        )

    def forward(self, side, x):
        for i in range(self.n_layers):
            x = self.layers[2 * i](x)
            mod = self.film_layers[i](side)
            layer_size = x.shape[-1]
            y = mod[:, : layer_size * layer_size]
            b = mod[:, -layer_size:]
            y = y.view(-1, layer_size, layer_size)
            x = (y @ x[:, :, None])[:, :, 0] + b
            x = self.layers[2 * i + 1](x)
        x = self.layers[-1](x)
        return x
