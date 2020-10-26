import math

import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal

from utils import TNP, TNM


class Layer(nn.Module):
    r"""
    Abstract Class for a Layer of a PGM

    Args:
        name (String): Name of the layer
    """

    def __init__(self, name="layer0"):
        super(Layer, self).__init__()
        self.name = name
        self.full_name = f"Abstract_{name}"
        self.shape = None

    def forward(self, *args):
        pass

    def gamma(self, *args):
        pass


class DReLULayer(Layer):
    r"""
    Layer of independent dReLU neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """

    def __init__(self, N=100, name="layer0"):
        super(DReLULayer, self).__init__(name)
        self.full_name = f"dReLU_{name}"
        self.N, self.shape = N, N
        self.phi = None
        self.params = nn.ParameterList([nn.Parameter(torch.tensor(1.), requires_grad=True),
                                        nn.Parameter(torch.tensor(1.), requires_grad=True),
                                        nn.Parameter(torch.tensor(0.), requires_grad=True),
                                        nn.Parameter(torch.tensor(0.), requires_grad=True)])

    def sample(self, probas, beta=1):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.N) for p in probas]).clamp(-5, 5)
        self.phi = phi
        _, _, p_plus, p_minus = self._Z(phi)
        sample_plus = TNP((phi - theta_plus) / gamma_plus, 1 / gamma_plus)
        sample_minus = TNM((phi - theta_minus) / gamma_minus, 1 / gamma_minus)
        return p_plus * sample_plus + p_minus * sample_minus

    def forward(self, h):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        h_plus = F.relu(h)
        h_minus = -F.relu(-h)
        return (h_plus.pow(2).mul(gamma_plus / 2) + h_minus.pow(2).mul(gamma_minus / 2)
                + h_plus.mul(theta_plus) + h_minus.mul(theta_minus)).sum(-1)

    def gamma(self, iv):
        z_plus, z_minus, _, _ = self._Z(iv)
        return torch.log((z_plus + z_minus).clamp(1e-8, 2e4)).sum(-1)

    def _Z(self, x):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        r_gamma_plus, r_gamma_minus = math.sqrt(gamma_plus), math.sqrt(gamma_minus)
        z_plus = (DReLULayer._phi(-(x - theta_plus) / r_gamma_plus) / r_gamma_plus)
        z_minus = (DReLULayer._phi((x - theta_minus) / r_gamma_minus) / r_gamma_minus)
        return z_plus, z_minus, DReLULayer.fillna(z_plus / (z_plus + z_minus)), DReLULayer.fillna(
            z_minus / (z_plus + z_minus))

    @staticmethod
    def _phi(x):
        r2 = math.sqrt(2)
        rpi2 = math.sqrt(math.pi / 2)
        phix = torch.exp(x ** 2 / 2) * torch.erfc(x / r2) * rpi2

        idx = (x > 5)
        phix[idx] = (1 / x[idx]) - (1 / x[idx] ** 3) + 3 / x[idx] ** 5

        idx = (x < - 5)
        phix[idx] = torch.exp(x[idx] ** 2 / 2) * rpi2
        return phix

    @staticmethod
    def fillna(x, val=1):
        idx = torch.where(x.__ne__(x))[0]
        x[idx] = val
        return x

class GaussianLayer(Layer):
    r"""
    Layer of independent Gaussian neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """

    def __init__(self, N=100, name="layer0"):
        super(GaussianLayer, self).__init__(name)
        self.full_name = f"Gaussian_{name}"
        self.N, self.shape = N, N

    def sample(self, probas, beta=1):
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.N) for p in probas])
        distribution = Normal(phi, 1)
        return distribution.sample()

    def forward(self, h):
        return -(h.pow(2) / 2).sum(-1)

    def gamma(self, Iv):
        return (Iv.pow(2) / 2).sum(-1)


class OneHotLayer(Layer):
    r"""
    Layer of independent One Hot neurons

    Args:
        weights (torch.FloatTensor): initial weights
        N (Integer): Number of neurons
        q (Integer): Number of values the neuron can take
        name (String): Name of the layer
    """

    def __init__(self, weights=None, N=100, q=21, name="layer0"):
        super(OneHotLayer, self).__init__(name)
        self.full_name = f"OneHot_{name}"
        self.N, self.q, self.shape = N, q, N * q
        self.linear = nn.Linear(self.shape, 1, bias=False)
        self.phi = None
        if weights is not None:
            self.linear.weights = weights.view(1, -1)
        for param in self.parameters():
            param.requires_grad = False

    def get_weights(self):
        return self.linear.weights

    def sample(self, probas, beta=1):
        batch_size = probas[0].size(0)
        phi = beta * sum([p.view(batch_size, self.q, self.N) for p in probas])
        phi += self.linear.weights.view(1, self.q, self.N)
        self.phi = phi
        distribution = OneHotCategorical(probs=F.softmax(phi, 1).permute(0, 2, 1))
        return distribution.sample().permute(0, 2, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.linear(x)

    def l2_reg(self):
        w = self.get_weights()
        return (w ** 2).sum()

