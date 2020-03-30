import numpy as np
import math

import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.distributions.bernoulli import Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal

from .metrics import TNP, TNM
from .utils import device

class Layer(nn.Module):
    r"""
    Abstract Class for a Layer of a PGM

    Args:
        name (String): Name of the layer
    """
    def __init__(self, name = "layer0"):
        super(Layer, self).__init__()
        self.name = name
        self.full_name = f"Abstract_{name}"
        self.shape = None

    def forward(self, x, *args, **kwargs):
        pass
        
    def sample(self, probas):
        pass
    
class dReLULayer(Layer):
    r"""
    Layer of independant dReLU neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """
    def __init__(self, N = 100, name = "layer0"):
        super(dReLULayer, self).__init__(name)
        self.full_name = f"dReLU_{name}"
        self.N, self.shape = N, N
        self.params = [torch.tensor(1., requires_grad = False), 
                                       torch.tensor(1., requires_grad = False), 
                                        torch.tensor(0., requires_grad = False), 
                                        torch.tensor(0., requires_grad = False)]
        
    def sample(self, probas): 
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        batch_size = probas[0].size(0)
        phi = sum([p.view(batch_size, self.N) for p in probas]).clamp(-5,5)
        _, _, p_plus, p_minus = self._Z(phi)
        sample_plus = TNP((phi - theta_plus)/gamma_plus, 1/gamma_plus)
        sample_minus = TNM((phi - theta_minus)/gamma_minus, 1/gamma_minus)
        return p_plus*sample_plus + p_minus*sample_minus
    
    def forward(self, h):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        h_plus = F.relu(h)
        h_minus = -F.relu(-h)
        return (h_plus.pow(2).mul(gamma_plus/2)+h_minus.pow(2).mul(gamma_minus/2)
                +h_plus.mul(theta_plus)+h_minus.mul(theta_minus) ).sum(-1)
    
    def gamma(self, Iv): 
        Z_plus, Z_minus, _, _ = self._Z(Iv)
        return torch.log(Z_plus+Z_minus).sum(-1)
    
    def _Z(self, x):
        gamma_plus, gamma_minus, theta_plus, theta_minus = self.params
        r_gamma_plus, r_gamma_minus = math.sqrt(gamma_plus), math.sqrt(gamma_minus)
        Z_plus = self._phi(-(x-theta_plus)/r_gamma_plus)/r_gamma_plus
        Z_minus = self._phi((x-theta_minus)/r_gamma_minus)/r_gamma_minus
        return Z_plus, Z_minus, Z_plus/(Z_plus+Z_minus), Z_minus/(Z_plus+Z_minus)
        
    def _phi(self, x):
        r2 = math.sqrt(2)
        rpi2 = math.sqrt(math.pi/2)
        return torch.exp(x**2/2)*(1-torch.erf(x/r2))*rpi2
    
class GaussianLayer(Layer):
    r"""
    Layer of independant Gaussian neurons

    Args:
        N (Integer): Number of neurons
        name (String): Name of the layer
    """
    def __init__(self, N = 100, name = "layer0"):
        super(GaussianLayer, self).__init__(name)
        self.full_name = f"Gaussian_{name}"
        self.N, self.shape = N, N
        
    def sample(self, probas): 
        batch_size = probas[0].size(0)
        phi = sum([p.view(batch_size, self.N) for p in probas])
        self.distribution = Normal(phi, 1)
        return self.distribution.sample()
    
    def forward(self, h):
        return (h.pow(2)/2).sum(-1)
    
    def gamma(self, Iv): 
        return (Iv.pow(2)/2).sum(-1)
        
class OneHotLayer(Layer):
    r"""
    Layer of independant One Hot neurons

    Args:
        weights (torch.FloatTensor): initial weights
        N (Integer): Number of neurons
        q (Integer): Number of values the neuron can take
        name (String): Name of the layer
    """
    def __init__(self, weights = None, N = 100, q = 21, name = "layer0"):
        super(OneHotLayer, self).__init__(name)
        self.full_name = f"OneHot_{name}"
        self.N, self.q, self.shape = N, q, N * q
        self.linear = nn.Linear(self.shape, 1)
        if weights is not None:
            self.linear.weights = weights.view(1,-1)

        
    def sample(self, probas):
        batch_size = probas[0].size(0)
        phi = sum([p.view(batch_size, self.q, self.N) for p in probas])
        phi += self.linear.weights.view(1, self.q, self.N)
        self.distribution = OneHotCategorical(probs = F.softmax(phi, 1).permute(0, 2, 1))
        return self.distribution.sample().permute(0, 2, 1)
    
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.linear(x)