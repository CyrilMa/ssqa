import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import math
 
# Loss

def kld_loss(recon_x, x, mu):
    BCE = F.cross_entropy(recon_x, x.argmax(1), reduction= "mean")
    KLD = -0.5 * torch.sum(- mu.pow(2))
    return BCE + KLD

# EBM loss

def hinge_loss(model, x, y, m = 1):
    e = model(x) 
    e_bar = torch.min((e + 1e9*y), 1, keepdim=True).values.view(e.size(0), 1, 
                                                                 e.size(-1))
    loss = F.relu(m+(e-e_bar)*y)
    return loss.sum()/(e.size(0))

def likelihood_loss(model, x, y, m = 1):
    e = model(x) 
    return (-F.log_softmax(F.log_softmax(-e,1), 1)*y).sum()

# Metrics

def aa_acc(x, recon_x):
    r"""
    Evaluate the ratio of amino acids retrieved in the reconstructed sequences

    Args:
        x (torch.Tensor): true sequence(s)
        recon_x (torch.Tensor): reconstructed sequence(s)
    """
    empty = torch.max(x, 1)[0].view(-1)
    x = torch.argmax(x, 1).view(-1)
    recon_x = torch.argmax(recon_x, 1).view(-1)
    return (((x==recon_x) * (empty!=0)).int().sum().item())/((empty!=0).int().sum().item())
    
# Regularizer

def l1b_reg(edge):
    r"""
    Evaluate the L1b factor for the weights of an edge

    Args:
        edge (Edge): edge to be evaluated
    """
    w = edge.get_weights()
    reg = torch.abs(w).sum(-1)
    return (reg**2).sum(0)

def l1_reg(edge):
    w = edge.get_weights()
    return torch.abs(w).sum()

def l2_reg(edge):
    w = edge.get_weights()
    return (w**2).sum()

def msa_mean(x, w):
    return (w*x).sum(0)/w.sum(0)

# Probabilistic law

inf = float("Inf")
r2 = math.sqrt(2)

def phi(x):
    return (1+torch.erf(x/r2))/2

def phi_inv(x):
    return r2*(torch.erfinv(2*x-1))

def TruncatedNormal(mu, sigma, a, b):
    x = torch.rand_like(mu)
    phi_a, phi_b = phi((a-mu)/sigma), phi((b-mu)/sigma)
    return phi_inv(phi_a + x*(phi_b - phi_a))*sigma+mu

def TNP(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = phi(-mu/sigma), 1
    return (phi_inv(phi_a + x*(phi_b - phi_a))*sigma+mu).clamp(0, 100)

def TNM(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = 0, phi(-mu/sigma)
    return (phi_inv(phi_a + x*(phi_b - phi_a))*sigma+mu).clamp(-100, 0)

# Gauges

def ZeroSumGauge(N = 31, q = 21):
    gauge = torch.eye(q*N)
    for i in range(q*N):
        gauge[i, (i+N*np.arange(q))%(q*N)] -= 1/q
    return gauge

