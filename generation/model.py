import torch
from torch import nn, optim
from torch.nn import functional as F


class Entropy(nn.Module):
    def __init__(self, q, N, h):
        super(Entropy, self).__init__()
        self.bilinear = nn.Bilinear(q * N, h, 1)

    def forward(self, x, h):
        return self.bilinear(x, h)


class Generator(nn.Module):
    def __init__(self, q, N, z_dim):
        super(Generator, self).__init__()
        self.q, self.N = q, N
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, q * N)

    def forward(self, h):
        x = self.linear(h).view(-1, self.q, self.N)
        return F.softmax(x, 1).view(-1, self.q * self.N)


class Energy(nn.Module):
    def __init__(self, ebm, q, N, matcher=None):
        super(Energy, self).__init__()
        self.ebm = ebm
        self.q, self.N = q, N
        self.matcher = matcher
        self.use_ssqa = False

    def forward(self, x):
        d = {"sequence": x}
        seq = -(self.ebm.integrate_likelihood(d, "hidden") / self.N - self.ebm.Z)
        if self.matcher and self.use_ssqa:
            struct = -self.matcher(x)
        else:
            struct = torch.tensor(0.)
        return seq + struct, seq.mean(), struct.mean()