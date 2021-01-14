import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

from ssqa import Matching
from utils import *
from torch.distributions.one_hot_categorical import OneHotCategorical

class Sampler(object):
    def  __init__(self):
        super(Sampler, self).__init__()

    def sample(self, data):
        pass


class IndependantSampler(Sampler):
    def __init__(self, p):
        super(IndependantSampler, self).__init__()
        self.sampler = OneHotCategorical(p)
        self.p = p

    def sample(self, data):
        n = data.size(0)
        return self.sampler.sample_n(n).permute(0, 2, 1)


class NaturalSampler(Sampler):
    def __init__(self):
        super(NaturalSampler, self).__init__()

    def sample(self, data):
        return data.permute(0, 2, 1)


class PGMSampler(Sampler):
    def __init__(self, model, visible_layers, hidden_layers, k=30):
        super(PGMSampler, self).__init__()
        self.model = model
        self.visible_layers = visible_layers
        self.hidden_layers = hidden_layers
        self.k = k

    def sample(self, data):
        d_0 = {k: v.float().to(device) for k, v in zip(LAYERS_NAME, [data]) if
               k in self.visible_layers}
        _, d_f = self.model.gibbs_sampling(d_0, self.visible_layers, self.hidden_layers, k=self.k)
        return d_f["sequence"]


class RejectionSampler(object):
    def __init__(self, loader, ssqa, scalers):
        super(RejectionSampler, self).__init__()
        self.dp3, self.dp8, self.pm3, self.pm8 = matchers
        self.scalerdp, self.scalerpm = scalers
        self.loader = loader

    def sample(self, sampler, n_samples, thr):
        samples = []
        while True:
            for batch_idx, data in enumerate(self.loader):
                print(f"{len(samples)}/{n_samples} [{int(100 * len(samples) / n_samples)}%]", end="\r")
                x = sampler.sample(data)
                ss
                u = np.array(list(range(30))).reshape(1, 1, 30)
                L = torch.cat([L3, L8], 1).clamp(1e-8, 1)
                p = ((L * u).sum(-1) - L_0)

                pm = ((-p - min_) / max_).mean(1)
                for x_, pm_ in zip(x, pm):
                    if len(samples) >= n_samples:
                        return samples
                    if pm_ > thr:
                        samples.append((x_, pm_))