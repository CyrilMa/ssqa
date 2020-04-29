from itertools import product
import math
import networkx as nx

import torch
from torch import nn, optim
from torch.nn import functional as F

from .edge import Edge
from .metrics import ZeroSumGauge
from .graphic import draw_G
from .utils import device


class MRF(nn.Module):
    r"""
    Class to handle Markov Random Field : graph of layers and edges.

    Args:
        layers (Dict): Keys are the name of the layers, values are the layers 
        edges (List of tuples): List of all edges between layers
    """

    def __init__(self, layers, edges):
        super(MRF, self).__init__()
        self.layers = nn.ModuleDict(layers)
        self.in_, self.out_ = [k for k in self.layers.keys() if k != "hidden"], "hidden"
        self.edges_name = edges
        self.edges = nn.ModuleDict({f"{u} -> {v}": Edge(layers[u], layers[v],
                                                        ZeroSumGauge(layers[u].N, layers[u].q)).to(device) for u, v in
                                    edges})
        self.G = self.build_graph()
        self.Z = 0; self.ais()
        draw_G(self.G)

    def build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(list(self.layers.keys()))
        G.add_edges_from(self.edges_name)
        return G

    def get_edge(self, i, o):
        return self.edges[f"{i} -> {o}"]
    
    def is_edge(self, i, o):
        return f"{i} -> {o}" in self.edges.keys()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def forward(self, d):
        return self.full_likelihood(d)

    def integrate_likelihood(self, d, out_lay, beta = 1):  # compute P(visible)
        out = self.layers[out_lay]
        ll = None
        for name, x in d.items():
            if name == out_lay:
                continue
            Ix = self.get_edge(name, out_lay)(x, False)
            gammaIx = out.gamma(beta * Ix)
            ll = gammaIx if ll is None else ll + gammaIx
            ll += self.layers[name](x).reshape(-1)
        return ll

    def full_likelihood(self, d, beta = 1):
        e = None
        for (in_, out_), edge in zip(self.edges_name, self.edges.values()):
            if in_ not in d.keys() or out_ not in d.keys():
                continue
            x, h = d[in_], d[out_]
            e = beta * (edge(x, False) * h).sum(-1).view(-1) if e is None else e + beta *(edge(x, False) * h).sum(-1).view(-1)
        for in_, x in d.items():
            e += self.layers[in_](x).view(-1)
        return e

    def gibbs_sampling(self, d, in_lays, out_lays, k=1, beta=1):
        lays = in_lays + out_lays
        for lay in out_lays:
            d[lay] = self._gibbs(d, lay, beta)
        d_0 = d.copy()
        for _ in range(k):
            for lay in lays:
                d[lay] = self._gibbs(d, lay, beta)
        return d_0, d
    
    
    def init_sample(self, n_samples, beta = 0):
        in_, out_ = self.in_, self.out_
        d = dict()
        out_layer = self.layers[out_]
        d[out_] = out_layer.sample([torch.zeros(n_samples, out_layer.N)], beta)
        for lay in in_:
            d[lay] = self._gibbs(d, lay)
        return d

    def ais(self, n_samples = 20, n_inter = 2000, verbose = False):
        in_, out_ = self.in_, self.out_
        d = self.init_sample(n_samples)
        betas = torch.linspace(0, 1, n_inter)
        weights = 0
        for i, (last_beta,beta) in enumerate(zip(betas[:-1], betas[1:])):
            _,d = self.gibbs_sampling(d, in_, [out_], 1, last_beta)
            weights += (self.full_likelihood(d, beta) - self.full_likelihood(d, last_beta)).mean(0).item()
            if verbose and not i % (n_inter//10):
                print(f"Iteration {i} : {weights}")
        Z_0 = 1/2 * self.layers[out_].N * math.log(2*math.pi)
        Z_0 += sum(self.layers[i].linear.weights.view(self.layers[i].q,-1).logsumexp(0).sum(0) for i in in_)
        Z = (weights + Z_0)/sum(self.layers[i].N for i in in_)
        if verbose:
            print(f"Estimated Z : {Z:.3f}")
        self.Z = Z
        return Z.item()


    def _gibbs(self, d, out_lay, beta=1):
        probas = []
        out_layer = self.layers[out_lay]
        for name, lay in d.items():
            if name == out_lay:
                continue
            distribution = self._distribution(name, out_lay, lay)
            if distribution is not None:
                probas.append(distribution)
        return out_layer.sample(probas, beta)
    

    def _distribution(self, i, o, x):
        if self.is_edge(i, o):
            return self.get_edge(i, o).forward(x, sample=False)
        if self.is_edge(o, i):
            return self.get_edge(o, i).backward(x, sample=False)
        return None
