from __future__ import annotations

import collections
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList
from src.module.FCLayers import FCLayers

def _identity(x):
    return x

class ExprLibrarySizeEncoder(torch.nn.Module):
    """Library size encoder."""

    def __init__(
        self,
        n_input: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
            **kwargs,
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, 1), torch.nn.LeakyReLU())

    def forward(self, x:torch.Tensor, *cat_list:int):
        return self.output(self.px_decoder(x, *cat_list))

## same as AccDecoder class
class AccLibrarySizeEncoder(nn.Module): 

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
            **kwargs,
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid())

    def forward(self, z: torch.Tensor, *cat_list: int):
        """Forward pass."""
        x = self.output(self.px_decoder(z, *cat_list))
        return x

class Encoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden : int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        **kwargs
        ):
        super().__init__()

        self.distributon = distribution
        self.var_eps = var_eps
        self.encoder= FCLayers(
            n_in = n_input,
            n_out = n_hidden,
            n_cat_list=n_cat_list,
            n_hidden = n_hidden,
            n_layers = n_layers,
            dropout_rate = dropout_rate,
            **kwargs
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.var_activation = torch.exp if var_activation is None else var_activation
        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation
    
    def forward(self, x: torch.Tensor, *cat_list : int):
        
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())

        return q_m, q_v, latent  




