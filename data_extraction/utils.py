import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

I = (lambda x: x)
LAYERS_NAME = ["sequence", "pattern_matching", "transitions"]
PROFILE_HEADER = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  'M->M', 'M->I', 'M->D', 'I->M', 'I->I',
                  'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D')  # yapf: disable
AMINO_ACIDS = AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = AA_IDS = {k: i for i, k in enumerate(AA)}
AA_MAT = None
device = "cpu"
DATA = "/home/malbranke/data/"

# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = {0: "I",
                    1: "S",
                    2: "H",
                    3: "E",
                    4: "G",
                    5: "B",
                    6: "T",
                    7: "C"}

abc_codes = {"a": 0, "b": 1, "c": 2}
# Converter for the DSSP secondary pattern_matching elements
# to the classical ones
dssp_to_abc = {"I": "c",
               "S": "c",
               "H": "a",
               "E": "b",
               "G": "a",
               "B": "b",
               "T": "c",
               "C": "c"}

def ss8_to_ss3(x):
    if x <= 2:
        return 0
    if x >= 5:
        return 2
    return 1

# Numpy utils

def to_onehot(a, shape):
    if shape[0] is None:
        shape = len(a), shape[1]
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot

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

# Old (ss as boxes)

def overlap_to_one(x, p):
    x = x.expand(x.size(0), p.size(-1))
    inter_low = torch.max(x[0], p[0])
    inter_high = torch.min(x[1], p[1])
    union_low = torch.min(x[0], p[0])
    union_high = torch.max(x[1], p[1])

    return torch.clamp((inter_high - inter_low) / (union_high - union_low), 0, 1)

def to_bb(x):
    C = torch.tensor(range(x.size(-1))).view(1, 1, -1)
    z = x[:, :2].detach()
    z = z.round()
    return z

def overlap(ground_truth, predictions):
    union_low = torch.min(ground_truth[:, 0], predictions[:, 0])
    union_high = torch.max(ground_truth[:, 1], predictions[:, 1])
    inter_low = torch.max(ground_truth[:, 0], predictions[:, 0])
    inter_high = torch.min(ground_truth[:, 1], predictions[:, 1])
    return torch.clamp((inter_high - inter_low) / (union_high - union_low), 0, 1)


def L_box(ground_truth, predictions):
    # mask = (predictions[b_pos, 2:, p_pos].argmax(1) == ground_truth[b_pos, 2:, g_pos].argmax(1)).int().reshape(-1, 1)
    return F.smooth_l1_loss(predictions, ground_truth[:, :2], reduction="mean")


def L_conf(ground_truth, predictions, olap, b_pos, g_pos):
    return F.nll_loss(torch.log(predictions[b_pos, :, g_pos]), ground_truth[b_pos, :, g_pos].argmax(1), reduction="sum")


def to_seq(p, size):
    seq = np.zeros(size, dtype=int)
    for bbox in p.numpy().T[::-1]:
        seq[bbox[0]: bbox[1]] = bbox[2]
    return seq
