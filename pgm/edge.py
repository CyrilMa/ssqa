import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F

from .utils import device

class Edge(nn.Module):
    r"""
    Class to handle the original Edge of the RBM 

    Args:
        lay_in (Layer): First Layer of the edge (by convention the visible one when it applies) 
        lay_out (Layer): Second Layer of the edge (by convention the hidden one when it applies)
        gauge (torch.FloatTensor): A projector made to handle the gauge
        weights (torch.FloatTensor): pretrained weights
    """
    def __init__(self, lay_in, lay_out, gauge = None, weights = None):
        super(Edge, self).__init__()
        
        # Constants
        self.in_layer, self.out_layer = lay_in, lay_out
        self.freeze = True
        self.gauge = None
        if gauge is not None:
            self.gauge = gauge.detach()
            self.gauge = self.gauge.to(device)

        # Model
        in_shape, out_shape = lay_in.shape, lay_out.shape
        self.linear = nn.Linear(in_shape, out_shape, False)
        self.reverse = nn.Linear(out_shape, in_shape, False)
        self.bn1 = nn.BatchNorm1d(out_shape)
        if weights is not None:
            self.linear.weight = weights
        self.reverse.weight.data = self.linear.weight.data.t()
        
    def freeze(self):
        self.freeze = True
        
    def unfreeze(self):
        self.freeze = False
        
    def get_weights(self):
        if self.gauge is not None:
            return self.linear.weight.mm(self.gauge)
        return self.linear.weight
    
    def backward(self, h, sample = True):
        h = h.reshape(h.size(0), -1)
        p = self.reverse(h)
        if sample:
            x = self.in_layer.sample([p])
            return x
        return p
    
    def forward(self, x, sample = True):
        x = x.reshape(x.size(0), -1)
        W = self.get_weights()
        p = F.linear(x, W, self.linear.bias)
        if sample:
            h = self.out_layer.sample([p])
            return h
        return p
    
    def gibbs_step(self, x, sample = True):
        x = x.reshape(x.size(0), -1)
        mu = self.linear(x)
        h = mu
        if sample:
            h = self.out_layer.sample([mu])
        mut = self.reverse(h)
        x_rec = mut
        if sample:
            x_rec = self.in_layer.sample([mut])
        return x_rec, h, mut, mu
    
    def save(self, filename):
        torch.save(self, filename)        
