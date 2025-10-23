import torch
import torch.nn as nn
from torch.distributions import Normal, Poisson

from math import log

import torch
import torch.nn.functional as F
from torch.distributions import Poisson
from torch.nn.functional import one_hot

from collections import OrderedDict

class DomainSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_features, class_labels = ['sc', 'spatial', 'bulk'], eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(DomainSpecificBatchNorm1d, self).__init__()
        self.bns = nn.ModuleDict(
            {key: nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) for key in class_labels})

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        bn = self.bns[domain_label[0]]
        return bn(x)


class Encoder(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_latent, 
        dropout_rate=0.1, 
        dims=[1024, 128]
    ):
        super(Encoder, self).__init__()
        
        self.var_eps = 1e-4

        self.encoder = nn.Sequential(OrderedDict([
            ('Linear0', nn.Linear(n_in, dims[0])), 
            ('DSBN0', DomainSpecificBatchNorm1d(dims[0])),
            ('ReLU0', nn.ReLU()), 
            ('Dropout0', nn.Dropout(p=dropout_rate))
        ]))
        
        for i in range(len(dims)-1):
            self.encoder.add_module(
                'Linear' + str(i+1), 
                nn.Linear(dims[i], dims[i+1])
            )
            self.encoder.add_module(
                'DSBN' + str(i+1), 
                DomainSpecificBatchNorm1d(dims[i+1])
            )
            self.encoder.add_module(
                'ReLU' + str(i+1),
                nn.ReLU()
            )
            self.encoder.add_module(
                'Dropout' + str(i+1), 
                nn.Dropout(p=dropout_rate)
            )

        self.mean_encoder = nn.Linear(dims[len(dims)-1], n_latent) # \mu
        self.logvar_encoder = nn.Linear(dims[len(dims)-1], n_latent) # \logvar
    
    def forward(self, x, tech_label):
        for layer in self.encoder:
            if isinstance(layer, DomainSpecificBatchNorm1d):
                x = layer(x, tech_label)
            else:
                x = layer(x)
        q_mu = self.mean_encoder(x)
        q_logvar = self.logvar_encoder(x)
        normal_dict = Normal(q_mu, q_logvar.div(2).exp() + 1e-17)
        latent = normal_dict.rsample()
        return q_mu, q_logvar, latent


class Decoder(nn.Module):
    def __init__(
        self, 
        n_latent, 
        n_out, 
        dims=[128, 1024]
    ):
        super(Decoder, self).__init__()

        self.px_decoder = nn.Sequential(OrderedDict([
            ('Linear0', nn.Linear(n_latent, dims[0])), 
            ('DSBN0', DomainSpecificBatchNorm1d(dims[0])),
            ('ReLU0', nn.ReLU()), 
            ('Dropout0', nn.Dropout(p=0.1))
        ]))

        for i in range(len(dims)-1):
            self.px_decoder.add_module(
                'Linear' + str(i+1), 
                nn.Linear(dims[i], dims[i+1])
            )
            self.px_decoder.add_module(
                'DSBN' + str(i+1), 
                DomainSpecificBatchNorm1d(dims[i+1])
            )
            self.px_decoder.add_module(
                'ReLU' + str(i+1),
                nn.ReLU()
            )
            self.px_decoder.add_module(
                'Dropout' + str(i+1), 
                nn.Dropout(p=0.1)
            )
        
        # lambda of Poisson
        self.st_px_scale_decoder = nn.Linear(dims[len(dims)-1], n_out)
        self.st_px_scale_decoder_activation = nn.Softmax(dim=-1)
        
        self.bulk_px_scale_decoder = nn.Linear(dims[len(dims)-1], n_out)
        self.bulk_px_scale_decoder_activation = nn.Softmax(dim=-1)
    
    def forward(self, x, tech_label):
        for layer in self.px_decoder:
            if isinstance(layer, DomainSpecificBatchNorm1d):
                x = layer(x, tech_label)
            else:
                x = layer(x)
        
        px_scale = self.bulk_px_scale_decoder(x)
        px_scale = self.bulk_px_scale_decoder_activation(px_scale)
        
        return px_scale


class comVI(nn.Module):
    def __init__(
        self, 
        n_in, 
        n_latent, 
        dropout_rate=0.1, 
        gene_likelihood='poisson', # 'zipoisson', 'poisson', 
        encoder_dims=[1024, 128],
        decoder_dims=[128, 1024], 
        library_scale=1e4
    ):
        super(comVI, self).__init__()
        
        self.eps = 1e-8
        self.gene_likelihood = gene_likelihood
        self.library_scale = library_scale
        
        self.encoder = Encoder(
            n_in=n_in, 
            n_latent=n_latent, 
            dropout_rate=dropout_rate, 
            dims=encoder_dims
        )
        self.decoder = Decoder(
            n_latent=n_latent, 
            n_out=n_in, 
            dims=decoder_dims
        )
        
    def forward(self, x, tech_label):
        library = torch.log(x.sum(1)).unsqueeze(1)
        
        x = torch.log(1 + x)
        
        q_mu, q_logvar, latent = self.encoder(x, tech_label)
        px_scale = self.decoder(latent, tech_label)

        zi_poisson = Poisson(torch.exp(library) * px_scale + self.eps)
        
        return {'z_mu': q_mu, 'z_logvar': q_logvar, 'latent': latent, 
                'px_scale': px_scale, 'library': library, 
                'zi_poisson': zi_poisson}
    
    def get_latent(self, x, tech_label):
        x = torch.log(1 + x)
        latent, _, _ = self.encoder(x, tech_label)
        return latent


class WeibullActivation(nn.Module):
    def __init__(
        self, 
        init_alpha = 1.0, 
        max_beta_value = 5.0, 
        scalefactor = 1.0
    ):
        super(WeibullActivation, self).__init__()
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        self.scalefactor = scalefactor
        
        self.bactive = nn.Softplus()
    
    def forward(self, x):
        a, b  = torch.split(x, 1, dim=-1)
        a = self.init_alpha * torch.exp(a) * self.scalefactor
        b = self.bactive(b)
        return torch.cat((a, b), -1)

class Weibull(nn.Module):
    def __init__(
        self, 
        n_in, 
        dropout_rate=0.1, 
        init_alpha = 1.0, 
        max_beta_value = 5
    ):
        super(Weibull, self).__init__()
        
        self.FCLayers = nn.Sequential(
            nn.Linear(n_in, 16), 
            nn.LayerNorm(16, elementwise_affine=False),
            nn.Tanh(),
            nn.Linear(16, 2)
        )
        
        self.activation = WeibullActivation(
            init_alpha=init_alpha, 
            max_beta_value=max_beta_value
        )
    
    def forward(self, x):
        for layer in self.FCLayers:
            x = layer(x)
        x = self.activation(x)
        
        return {'alpha': x[:, 0], 
                'beta': x[:, 1]}
    

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.sum(XX + YY - XY -YX)
        return loss


def KL_Divergence(mu, logvar):
    #return torch.maximum(-0.5 * (logvar + 1 - mu.pow(2) - logvar.exp()), torch.tensor(0.1)).mean()
    return -0.5 * (logvar + 1 - mu.pow(2) - logvar.exp())

def loss_func(
        output, x, tech_label, 
        gene_likelihood='poisson',# 'zipoisson', 'poisson'
        reconst_weight = 1.0, 
        gaussian_weight = 1.0
    ):

    # z normalization
    kl_divergence_z = KL_Divergence(output['z_mu'], output['z_logvar']).sum(dim=1).sum()#.mean()
    
    # reconstruction loss
    x.to(torch.int32)
    if gene_likelihood == 'poisson':
        reconst_loss = -output['zi_poisson'].log_prob(x).sum(dim=1).sum()#.mean()
    elif gene_likelihood == 'zipoisson':
        reconst_loss = -output['zi_poisson'].zi_log_prob(x).sum(dim=1).sum()#.mean()

    
    return {'total_loss': gaussian_weight * kl_divergence_z + reconst_weight * reconst_loss, 
            'normal_loss': kl_divergence_z * gaussian_weight, 
            'reconstruction_loss': reconst_loss * reconst_weight}


def continuous_weibull_loglik(x, c, alpha, beta, clip_prob=1e-6):
    eps = 1e-16
    xa = (x + eps) / alpha
    loglik = c * (torch.log(beta) + beta * torch.log(xa)) - torch.pow(xa, beta)
    if clip_prob is not None:
        loglik = torch.clamp(loglik, log(clip_prob), log(1 - clip_prob))
    return -1.0 * torch.sum(loglik)

def weibull_qf(p, alpha, beta, device='cuda'):
    return alpha * ((-torch.log(torch.tensor(1 - p))).to(device) ** (1 / beta))

def spatial_regulation(
    features, 
    idx, 
    weight_matrix, 
    device='cuda',
    tech='spatial'
):
    if tech[0] != 'spatial':
        return torch.tensor(0.)
    distance_matrix = torch.cdist(features, features)
    sub_weight_matrix = weight_matrix[idx, :]
    sub_weight_matrix = sub_weight_matrix[:, idx]
    sub_weight_matrix = torch.from_numpy(sub_weight_matrix.toarray()).to(device)

    return torch.sum(distance_matrix * sub_weight_matrix)