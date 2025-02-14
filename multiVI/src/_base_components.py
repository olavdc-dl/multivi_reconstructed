from __future__ import annotations

import collections
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.nn._utils import ExpActivation

def _identity(x):
    return x

class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        inject_covariates: bool = True,
        bias : bool = True, 
        activation_fn: nn.Module = nn.ReLU,
        ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        # batch correction
        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        
        cat_dim = sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i) ,
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def forward(self, x: torch.Tensor, *cat_list: int):
        
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1).to(dtype=torch.int64), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]

        # forward pass of the layers
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            if (
                                x.device.type == "mps"
                            ):  # TODO: remove this when MPS supports for loop.
                                x = torch.cat(
                                    [(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0
                                )
                            else:
                                x = torch.cat(
                                    [layer(slice_x).unsqueeze(0) for slice_x in x], dim=0
                                )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list 
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                
                        x = layer(x)
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

class DecoderSCVI(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs):
        super().__init__()

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout 
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)
    
    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):

        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout

class Decoder(nn.Module): 

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

class Classifier(nn.Module):
    """Basic fully-connected NN classifier.

    Parameters
    ----------
    n_input
        Number of input dimensions
    n_hidden
        Number of nodes in hidden layer(s). If `0`, the classifier only consists of a
        single linear layer.
    n_labels
        Numput of outputs dimensions
    n_layers
        Number of hidden layers. If `0`, the classifier only consists of a single
        linear layer.
    dropout_rate
        dropout_rate for nodes
    logits
        Return logits or not
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    activation_fn
        Valid activation function from torch.nn
    **kwargs
        Keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 128,
        n_labels: int = 5,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        logits: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        layers = []

        if n_hidden > 0 and n_layers > 0:
            layers.append(
                FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            )
        else:
            n_hidden = n_input

        layers.append(nn.Linear(n_hidden, n_labels))

        if not logits:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """Forward computation."""
        return self.classifier(x)
