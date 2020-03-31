from itertools import product
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
        self.edges_name = edges
        self.edges = nn.ModuleDict({f"{u} -> {v}": Edge(layers[u], layers[v],
                                                        ZeroSumGauge(layers[u].N, layers[u].q)).to(device) for u, v in
                                    edges})
        self.G = self.build_graph()
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

    def integrate_likelihood(self, d, out_lay):  # compute P(visible)
        out = self.layers[out_lay]
        ll = None
        for name, x in d.items():
            if name == out_lay:
                continue
            Ix = self.get_edge(name, out_lay)(x)
            gammaIx = out.gamma(Ix)
            ll = gammaIx if ll is None else ll + gammaIx
            ll += self.layers[name](x).reshape(-1)
        return ll

    def full_likelihood(self, d):
        e = None
        for (in_, out_), edge in zip(self.edges_name, self.edges.values()):
            if in_ not in d.keys() or out_ not in d.keys():
                continue
            x, h = d[in_], d[out_]
            e = (edge(x, False) * h).sum(-1).view(-1) if e is None else e + (edge(x, False) * h).sum(-1).view(-1)
        for in_, x in d.items():
            e += self.layers[in_](x).view(-1)
        return e

    def gibbs_sampling(self, d, in_lays, out_lays, k=1):
        lays = in_lays + out_lays
        for lay in out_lays:
            d[lay] = self._gibbs(d, lay)
        d_0 = d.copy()
        for _ in range(k):
            for lay in lays:
                d[lay] = self._gibbs(d, lay)
        return d_0, d

    def annealed_gibbs_sampling(self, x, in_, out_, beta, log_p0, k=10):
        for _ in range(k):
            distribution = (1 - beta) * self._distribution(in_, out_, x)
            h = out_.sample([distribution])
            distribution = (1 - beta) * self._distribution(out_, in_, h)
            x = in_.annealed_sample([distribution], log_p0)
        return x

    def _gibbs(self, d, out_lay):
        probas = []
        out_layer = self.layers[out_lay]
        for name, lay in d.items():
            if name == out_lay:
                continue
            distribution = self._distribution(name, out_lay, lay)
            if distribution is not None:
                probas.append(distribution(name, out_lay, lay))
        return out_layer.sample(probas)

    def _distribution(self, i, o, x):
        if self.is_edge(i, o):
            return self.get_edge(i, o).forward(x, sample=False)
        if self.is_edge(o, i):
            return self.get_edge(o, i).backward(x, sample=False)
        return None
