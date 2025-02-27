from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld
from torch.nn import functional as F

from src.distributions.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)

from src.module.encoders import ExprLibrarySizeEncoder, AccLibrarySizeEncoder, Encoder
from src.module.decoders import ExprDecoder, AccDecoder, ProDecoder, LinearDecoderSCVI
from src.module.FCLayers import FCLayers
from src.module.classifier import Classifier
from src.module._utils import masked_softmax

class LINEAR_MULTIVAE(torch.nn.Module):

    def __init__(
        self,
        n_hidden: int = None,
        n_latent: int = None,
        n_input_regions: int = 0,
        n_input_proteins: int = 0,
        n_input_genes: int = 0,
        modality_weights: Literal["equal", "cell", "universal"] = "equal",
        n_obs: int = 0,
        n_layers_encoder : int = 2,
        n_layers_decoder : int = 2,
        n_continuous_cov: int = 0,
        n_batch : int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = True,
        deeply_inject_covariates: bool = True,
        protein_dispersion: str = "protein",
        
    ):
        super().__init__()

        # INIT PARAMS
        self.n_input_regions = n_input_regions
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers_encoder =  n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.protein_dispersion = protein_dispersion

        self.gene_dispersion = gene_dispersion
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates

        self.n_batch = n_batch
        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        encoder_cat_list = cat_list if encode_covariates else None

        ### EXPRESSION ###

        # expression dispersion parameters
        self.gene_likelihood = gene_likelihood
        self.dropout_rate = dropout_rate

        # expression encoder
        self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))

        self.z_encoder_expression = Encoder(
            n_input = self.n_input_genes,
            n_output = self.n_latent,
            n_cat_list=encoder_cat_list,
            n_hidden = self.n_hidden,
            n_layers = self.n_layers_encoder,
            dropout_rate=self.dropout_rate,
            distribution = self.latent_distribution,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            activation_fn=torch.nn.LeakyReLU,
            )
        
        # Expression library size encoder
        self.l_encoder_expression = ExprLibrarySizeEncoder(
            n_input = self.n_input_genes,
            n_layers = self.n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden = self.n_hidden,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )
        
        # Expression decoder
        self.z_decoder_expression = LinearDecoderSCVI(
            n_input = self.n_latent,
            n_output = self.n_input_genes,
            n_cat_list= None ,
            use_batch_norm = True,
            use_layer_norm = False,
            bias=False,
        )

        # modality alignment
        self.n_obs = n_obs
        self.modality_weights = modality_weights
        self.n_modalities = int(n_input_genes > 0) + int(n_input_regions > 0)

        max_n_modalities = 3
        if modality_weights == "equal":
            mod_weights = torch.ones(max_n_modalities)
            self.register_buffer("mod_weights", mod_weights)
        elif modality_weights == "universal":
            self.mod_weights = torch.nn.Parameter(torch.ones(max_n_modalities))
        else:  # cell-specific weights
            self.mod_weights = torch.nn.Parameter(torch.ones(n_obs, max_n_modalities))


        ### ACCESSIBILITY ###

        # accessibility encoder
        if self.n_input_regions == 0:
            input_acc = 1
        else:
            input_acc = self.n_input_regions
        n_input_encoder_acc = input_acc + n_continuous_cov * encode_covariates
    
        self.z_encoder_accessibility = Encoder(
            n_input=n_input_encoder_acc,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        self.z_decoder_accessibility = AccDecoder(
            n_input=self.n_latent + self.n_continuous_cov,
            n_output=n_input_regions,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

        # accessibility library size encoder
        self.l_encoder_accessibility = AccLibrarySizeEncoder(
            n_input=n_input_encoder_acc,
            n_output=1,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            n_layers=self.n_layers_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))
        
        ## PROTEIN
        if self.n_input_proteins == 0:
            input_pro = 1

        n_input_encoder_pro = input_pro + n_continuous_cov * encode_covariates

        self.z_encoder_protein = Encoder(
            n_input=n_input_encoder_pro,
            n_layers=self.n_layers_encoder,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU,
            distribution=self.latent_distribution,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
        )

        # protein decoder
        self.z_decoder_pro = ProDecoder(
            n_input=self.n_latent,
            n_output_proteins=n_input_proteins,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
        )

        # protein dispersion parameters
        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_labels))
        else:  # protein-cell
            pass

        ## ADVERSARIAL TRAINING
        self.n_output_classifier = self.n_batch
        self.adversarial_classifier = Classifier(
            n_input=self.n_latent,
            n_hidden=32,
            n_labels=self.n_output_classifier,
            n_layers=2,
            logits=True,
        )
        self.automatic_optimization = False

    def inference(
        self,
        x,
        y,
        batch_index,
        cell_idx,
        cat_covs = None,
        cont_covs = None,
        ) -> dict[str, torch.Tensor]:

        x_rna = x[:, : self.n_input_genes]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_atac.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        
        encoder_input_accessibility = x_atac
        encoder_input_expression = x_rna
        encoder_input_protein = y
         
        # Z Encoders
        qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )
        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
        qzm_pro, qzv_pro, z_pro = self.z_encoder_protein(
            encoder_input_protein, batch_index, *categorical_input
        )

        # L encoders
        libsize_expr = self.l_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input
        )
    
        libsize_acc = self.l_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input
        )

        # mix representations
        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)
    
        qz_m = mix_modalities(
            (qzm_expr, qzm_acc, qzm_pro), (mask_expr, mask_acc, mask_pro), weights
            )

        qz_v = mix_modalities(
            (qzv_expr, qzv_acc, qzv_pro),
            (mask_expr, mask_acc, mask_pro),
            weights,
            torch.sqrt,
        )

        # sample
        untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        z = self.z_encoder_accessibility.z_transformation(untran_z)

        outputs = {
            "x": x,
            "z": z,
            "qz_m" : qz_m,
            "qz_v" : qz_v,
            "qzm_expr" : qzm_expr,
            "qzv_expr" : qzv_expr,
            "qzm_acc" : qzm_acc,
            "qzv_acc" : qzv_acc,
            "qzm_pro": qzm_pro,
            "qzv_pro": qzv_pro,
            "libsize_expr": libsize_expr,
            "libsize_acc": libsize_acc,
            }

        return outputs
       
    def generative(
        self,
        z,
        batch_index,
        libsize_expr= None,
        cont_covs=None,
        cat_covs=None,):

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        latent  = z
        decoder_input = latent

        # Accessibility decoder
        p = self.z_decoder_accessibility(decoder_input, batch_index, *categorical_input)

        # Expression decoder
        px_scale, _ , px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input,
            libsize_expr,
            batch_index,
            *categorical_input
            )

        # Protein Decoder
        py_, log_pro_back_mean = self.z_decoder_pro(decoder_input, batch_index, *categorical_input)
        # Protein Dispersion
        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(F.one_hot(label.squeeze(-1), self.n_labels).float(), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        py_["r"] = py_r

        px_r = self.px_r
        px_r = torch.exp(px_r)

        return {
            "p" : p,
            "px_scale": px_scale,
            "px_r": torch.exp(self.px_r),
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "py_": py_,
        }

    def loss(
        self,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0):
       
        x = inference_outputs["x"]

        x_rna = x[:, : self.n_input_genes]
        x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_atac.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        # print(mask_expr)
        # print(mask_acc)
        # print(mask_pro)
    
        # ACCESSIBILITY
        p = generative_outputs["p"]
        libsize_acc = inference_outputs["libsize_acc"]
        rl_accessibility = self.get_reconstruction_loss_accessibility(x_atac, p, libsize_acc)

        # EXPRESSION
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # PROTEIN
        if mask_pro.sum().gt(0):
            py_ = generative_outputs["py_"]
            rl_protein = get_reconstruction_loss_protein(y, py_, None)
        else:
            rl_protein = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        # reconstruction loss
        rl_expression = self.get_reconstruction_loss_expression(
            x_rna, px_rate, px_r, px_dropout
        )
        recon_loss_expression = rl_expression * mask_expr
        recon_loss_accessibility = rl_accessibility * mask_acc
        recon_loss_protein = rl_protein * mask_pro
        recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_protein

        # Compute KLD between Z and N(0,I)
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        kl_div_z = kld(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)

        # Compute KLD between distributions for paired data
        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
            (inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
            mask_expr,
            mask_acc,
            mask_pro,
        )

        # KL WARMUP
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired

        loss = torch.mean(recon_loss + weighted_kl_local)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": recon_loss_protein,
        }

        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }

        return loss, kl_local, recon_losses

    def get_reconstruction_loss_expression(self, x, px_rate, px_r, px_dropout):
        """Computes the reconstruction loss for the expression data."""
        rl = 0.0
        if self.gene_likelihood == "zinb":
            rl = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            rl = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "poisson":
            rl = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return rl

    def get_reconstruction_loss_accessibility(self, x, p, d):
        """Computes the reconstruction loss for the accessibility data."""
        reg_factor = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        return torch.nn.BCELoss(reduction="none")(p * d * reg_factor, (x > 0).float()).sum(dim=-1)

    def _compute_mod_penalty(self, mod_params1, mod_params2, mod_params3, mask1, mask2, mask3):

        mask12 = torch.logical_and(mask1, mask2)
        mask13 = torch.logical_and(mask1, mask3)
        mask23 = torch.logical_and(mask3, mask2)

        pair_penalty = torch.zeros(mask1.shape[0], device=mask1.device, requires_grad=True)
        if mask12.sum().gt(0):
            penalty12 = sym_kld(
                mod_params1[0],
                mod_params1[1].sqrt(),
                mod_params2[0],
                mod_params2[1].sqrt(),
            )
            penalty12 = torch.where(mask12, penalty12.T, torch.zeros_like(penalty12).T).sum(
                dim=0
            )
            pair_penalty = pair_penalty + penalty12
        if mask13.sum().gt(0):
            penalty13 = sym_kld(
                mod_params1[0],
                mod_params1[1].sqrt(),
                mod_params3[0],
                mod_params3[1].sqrt(),
            )
            penalty13 = torch.where(mask13, penalty13.T, torch.zeros_like(penalty13).T).sum(
                dim=0
            )
            pair_penalty = pair_penalty + penalty13
        if mask23.sum().gt(0):
            penalty23 = sym_kld(
                mod_params2[0],
                mod_params2[1].sqrt(),
                mod_params3[0],
                mod_params3[1].sqrt(),
            )
            penalty23 = torch.where(mask23, penalty23.T, torch.zeros_like(penalty23).T).sum(
                dim=0
            )
            pair_penalty = pair_penalty + penalty23

        return pair_penalty

    def loss_adversarial_classifier(self, z, batch_index, predict_true_class=True):
        """Loss for adversarial classifier."""
        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))

        if predict_true_class:
            cls_target = torch.nn.functional.one_hot(batch_index.squeeze(-1).to(dtype=torch.int64), n_classes)
        else:
            one_hot_batch = torch.nn.functional.one_hot(batch_index.squeeze(-1).to(dtype=torch.int64), n_classes)
            # place zeroes where true label is
            cls_target = (~one_hot_batch.bool()).float()
            cls_target = cls_target / (n_classes - 1)

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss

    
def mix_modalities(Xs, masks, weights, weight_transform: callable = None):
    """Compute the weighted mean of the Xs while masking unmeasured modality values.

    Parameters
    ----------
    Xs
        Sequence of Xs to mix, each should be (N x D)
    masks
        Sequence of masks corresponding to the Xs, indicating whether the values
        should be included in the mix or not (N)
    weights
        Weights for each modality (either K or N x K)
    weight_transform
        Transformation to apply to the weights before using them
    """
    # (batch_size x latent) -> (batch_size x modalities x latent)
    Xs = torch.stack(Xs, dim=1)
    # (batch_size) -> (batch_size x modalities)
    masks = torch.stack(masks, dim=1).float()
    weights = masked_softmax(weights, masks, dim=-1)

    # (batch_size x modalities) -> (batch_size x modalities x latent)
    weights = weights.unsqueeze(-1)

    if weight_transform is not None:
        weights = weight_transform(weights)

    # sum over modalities, so output is (batch_size x latent)
    return (weights * Xs).sum(1)

def sym_kld(qzm1, qzv1, qzm2, qzv2):
    """Symmetric KL divergence between two Gaussians."""
    rv1 = Normal(qzm1, qzv1.sqrt())
    rv2 = Normal(qzm2, qzv2.sqrt())

    return kld(rv1, rv2) + kld(rv2, rv1)


def get_reconstruction_loss_protein(y, py_, pro_batch_mask_minibatch=None):
    """Get the reconstruction loss for protein data."""
    py_conditional = NegativeBinomialMixture(
        mu1=py_["rate_back"],
        mu2=py_["rate_fore"],
        theta1=py_["r"],
        mixture_logits=py_["mixing"],
    )

    reconst_loss_protein_full = -py_conditional.log_prob(y)

    if pro_batch_mask_minibatch is not None:
        temp_pro_loss_full = pro_batch_mask_minibatch.bool() * reconst_loss_protein_full
        rl_protein = temp_pro_loss_full.sum(dim=-1)
    else:
        rl_protein = reconst_loss_protein_full.sum(dim=-1)

    return rl_protein