import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F


# Torch utils

def trace(x, offset):
    shape = x.size()
    s1, s2 = shape[-2], shape[-1]
    x = x.reshape(-1, s1, s2)
    idxs = (torch.arange(s1)+offset)%s2
    return x[:, torch.arange(s1), idxs].sum(-1).view(*shape[:-2])

def mode(X, bins = 100):
    modes = []
    for x in X.t():
        idx = plt.hist(x, bins = bins)[0].argmax()
        modes.append(plt.hist(x, bins = bins)[1][idx])
    return torch.tensor(modes)

# Loss

def kld_loss(recon_x, x, mu):
    BCE = F.cross_entropy(recon_x, x.argmax(1), reduction="mean")
    KLD = -0.5 * torch.sum(- mu.pow(2))
    return BCE + KLD

def hinge_loss(model, x, y, m=1):
    e = model(x)
    e_bar = torch.min((e + 1e9 * y), 1, keepdim=True)[0].view(e.size(0), 1,
                                                              e.size(-1))
    loss = F.relu(m + (e - e_bar) * y)
    return loss.sum() / (e.size(0))

def likelihood_loss(model, x, y):
    e = model(x)
    return (-F.log_softmax(F.log_softmax(-e, 1), 1) * y).sum()

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
    return (((x == recon_x) * (empty != 0)).int().sum().item()) / ((empty != 0).int().sum().item())


# Regularizer

def msa_mean(x, w):
    return (w * x).sum(0) / w.sum(0)

# Probabilistic law

inf = float("Inf")
r2 = math.sqrt(2)
SAFE_BOUND = 1 - 1e-7

def phi(x):
    return (1 + torch.erf(x / r2)) / 2

def phi_inv(x):
    return r2 * torch.erfinv((2 * x - 1).clamp(-SAFE_BOUND, SAFE_BOUND))

def TNP(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = phi(-mu / sigma), torch.tensor(1.)
    a = (phi_a + x * (phi_b - phi_a))
    return phi_inv(a) * sigma + mu

def TNM(mu, sigma):
    x = torch.rand_like(mu)
    phi_a, phi_b = torch.tensor(0.), phi(-mu / sigma)
    a = (phi_a + x * (phi_b - phi_a))
    return phi_inv(a) * sigma + mu

def TruncatedNormal(mu, sigma, a, b):
    x = torch.rand_like(mu)
    phi_a, phi_b = phi((a - mu) / sigma), phi((b - mu) / sigma)
    a = (phi_a + x * (phi_b - phi_a))
    return phi_inv(a) * sigma + mu

# Gauges

def ZeroSumGauge(N=31, q=21):
    gauge = torch.eye(q * N)
    for i in range(q * N):
        gauge[i, (i + N * np.arange(q)) % (q * N)] -= 1 / q
    return gauge
