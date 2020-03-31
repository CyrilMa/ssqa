import math

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical


class AIS:

    def __init__(self, model, p_0, in_="visible", out_='hidden'):
        super(AIS, self).__init__()
        self.model = model
        self.q, self.N = p_0.size()
        self.in_, self.out_ = in_, out_
        self.p_0 = p_0.view(-1)
        self.log_p_0 = torch.log(self.p_0 + 1e-8)
        edge = model.get_edge(self.in_, self.out_)
        self.gammaI = lambda x: model.layers[self.out_].gamma(edge(x))
        self.W = model.layers[self.in_].linear.weights.view(-1)
        self.distribution = OneHotCategorical(p_0.t())

    def update(self, model):
        model = self.model
        edge = model.get_edge(self.in_, self.out_)
        self.gammaI = lambda x: model.layers[self.out_].gamma(edge(x))
        self.W = model.layers[self.in_].linear.weights.view(-1)

    def run(self, n_samples, n_inter, verbose=0):
        betas = torch.linspace(0, 1, n_inter)
        samples = self._sample(n_samples)
        logp0 = self.log_f0(samples)
        weights, last_beta = torch.zeros(n_samples), 0
        for i, beta in enumerate(betas[1:]):
            samples = self.model.annealed_gibbs_sampling(samples, self.in_, self.out_, beta,  self.log_p_0, 10)
            weights += self.log_fj(samples, beta) - self.log_fj(samples, last_beta)
            if verbose and not i % 100:
                print(f"Iteration {i}")
            last_beta = beta
        logpn = self.log_fn(samples)
        Z = ((logpn - logp0 - math.log(n_samples)).logsumexp(0)) / self.N
        print(f"Estimated Z : {Z:.3f}")
        return Z

    def _sample(self, n_samples):
        return self.distribution.sample((n_samples,)).permute(0, 2, 1).reshape(n_samples, -1)

    def log_f0(self, x):
        return (self.log_p_0 * x).sum(-1)

    def log_fn(self, x):
        ll = (x * self.W).sum(-1)
        ll += self.gammaI(x)
        return ll.detach()

    def log_fj(self, x, beta):
        return self.log_f0(x) * beta + self.log_fn(x) * (1 - beta)

    def probs(self, x):
        return (self.p_0.view(1, -1) * x).view(-1, self.q, self.N).permute(0, 2, 1)
