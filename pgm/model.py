import time
import networkx as nx

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .edge import Edge
from .utils import *
from .graphic import draw_G
from .utils import device

class MRF(nn.Module):
    r"""
    Class to handle Markov Random Field : graph of layers and edges.
    Args:
        layers (Dict): Keys are the name of the layers, values are the layers
        edges (List of tuples): List of all edges between layers
    """

    def __init__(self, layers, edges, name):
        super(MRF, self).__init__()
        self.layers = nn.ModuleDict(layers)
        self.in_, self.out_ = [k for k in self.layers.keys() if k != "hidden"], "hidden"
        self.edges_name = edges
        self.edges = nn.ModuleDict({f"{u} -> {v}": Edge(layers[u], layers[v],
                                                        ZeroSumGauge(layers[u].N, layers[u].q)).to(device) for u, v in
                                    edges})
        self.G = self.build_graph()
        self.Z = 0
        self.ais()
        self.device = "cpu"
        self.name = f"{name}-{self.layers[self.out].N}"
        self.writer = SummaryWriter(self.name)

        draw_G(self.G)

    def __repr__(self):
        return f"MRF {self.name}"

    def to(self, device):
        super(MRF, self).to(device)
        self.device = device
        return self

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

    def integrate_likelihood(self, d, out_lay, beta=1):  # compute P(visible)
        out = self.layers[out_lay]
        ll = None
        for name, x in d.items():
            if name == out_lay:
                continue
            Ix = self.get_edge(name, out_lay)(x, False)
            gammaIx = out.gamma(beta * Ix)
            ll = gammaIx if ll is None else ll + gammaIx
            ll += self.layers[name](x).reshape(-1)
        return ll + out.N / 2 * math.log(2 * math.pi)

    def full_likelihood(self, d, beta=1):
        e = None
        for (in_, out_), edge in zip(self.edges_name, self.edges.values()):
            if in_ not in d.keys() or out_ not in d.keys():
                continue
            x, h = d[in_], d[out_]
            e = beta * (edge(x, False) * h).sum(-1).view(-1) if e is None else e + beta * (edge(x, False) * h).sum(
                -1).view(-1)
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

    def init_sample(self, n_samples, beta=0):
        in_, out_ = self.in_, self.out_
        d = dict()
        out_layer = self.layers[out_]
        d[out_] = out_layer.sample([torch.zeros(n_samples, out_layer.N)], beta)
        for lay in in_:
            d[lay] = self._gibbs(d, lay, beta)
        return d

    def ais(self, n_samples=20, n_inter=2000, verbose=False):
        in_, out_ = self.in_, self.out_
        d = self.init_sample(n_samples)
        betas = torch.linspace(0, 1, n_inter)
        weights = 0
        for i, (last_beta, beta) in enumerate(zip(betas[:-1], betas[1:])):
            _, d = self.gibbs_sampling(d, in_, [out_], 1, last_beta)
            weights += (self.full_likelihood(d, beta) - self.full_likelihood(d, last_beta)).mean(0).item()
            if verbose and not i % (n_inter // 10):
                print(f"Iteration {i} : {weights}")
        Z_0 = 1 / 2 * self.layers[out_].N * math.log(2 * math.pi)
        Z_0 += sum(self.layers[i].linear.weights.view(self.layers[i].q, -1).logsumexp(0).sum(0) for i in in_)
        Z = (weights + Z_0) / sum(self.layers[i].N for i in in_)
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

    def write_tensorboard(self, logs, n_iter):
        for k, v in logs.items():
            self.writer.add_scalar(k, v, n_iter)

    def train_epoch(self, optimizer, loader, visible_layers, hidden_layers, gammas, epoch, savepath="seq100"):
        start = time.time()
        self.train()
        mean_loss, mean_reg, mean_acc = 0, 0, 0
        edges = [self.get_edge(v, "hidden") for v in visible_layers]
        for batch_idx, data in enumerate(loader):
            d_0 = {k: v.float().permute(0, 2, 1).to(device) for k, v in zip(LAYERS_NAME, data[:-2]) if k in visible_layers}
            w = data[-1].float().to(device)
            batch_size, q, N = d_0["sequence"].size()

            # Sampling
            d_0, d_f = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=10)

            # Optimization
            optimizer.zero_grad()
            e_0, e_f = self(d_0), self(d_f)
            loss = msa_mean(e_f - e_0, w)
            reg = torch.tensor(0.)
            for gamma, edge in zip(gammas, edges):
                reg += gamma * edge.l1b_reg()
            loss += reg
            loss.backward()
            optimizer.step()

            # Metrics
            d_0, d_f = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=1)
            acc = aa_acc(d_0["sequence"].view(batch_size, q, N), d_f["sequence"].view(batch_size, q, N))
            ll = msa_mean(self.integrate_likelihood(d_f, "hidden"), w) / N
            mean_loss = (mean_loss * batch_idx + ll.item()) / (batch_idx + 1)
            mean_reg = (mean_reg * batch_idx + reg) / (batch_idx + 1)
            mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60

            print(
            f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''',
            end="\r")
        print(
            f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''')
        logs = {"Train Loss" : mean_loss, "Train reg" : mean_reg, "Train acc" : mean_acc}
        self.write_tensorboard(logs, epoch)
        if not epoch % 30:
            print(
                f'''Train Epoch: {epoch} [100%] || Time: {m} min {s} || Loss: {mean_loss:.3f} || Reg: {mean_reg:.3f} || Acc: {mean_acc:.3f}''')
            self.save(f"{savepath}_{epoch}.h5")


    def val(self, loader, visible_layers, hidden_layers, epoch):
        start = time.time()
        self.eval()
        mean_pv, mean_pvh, mean_reg, mean_acc = 0, 0, 0, 0
        self.ais()
        for batch_idx, data in enumerate(loader):
            d_0 = {k: v.float().permute(0, 2, 1).to(device) for k, v in zip(LAYERS_NAME, data[:-1]) if k in visible_layers}
            w = data[-1].float().to(device)
            batch_size, q, N = d_0["sequence"].size()
            # Sampling
            d_0, d_f = self.gibbs_sampling(d_0, visible_layers, hidden_layers, k=10)

            acc = aa_acc(d_0["sequence"].view(batch_size, q, N), d_f["sequence"].view(batch_size, q, N))
            pv = msa_mean(self.integrate_likelihood(d_f, "hidden"), w)/N - self.Z
            pvh = msa_mean(self.full_likelihood(d_f), w)/N - self.Z
            mean_pv = (mean_pv * batch_idx + pv.item()) / (batch_idx + 1)
            mean_pvh = (mean_pvh * batch_idx + pvh.item()) / (batch_idx + 1)
            mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        logs = {"Val P(v)" : mean_pv, "Val P(v,h)" : mean_pvh, "Val acc" : mean_acc, "AIS": self.Z}
        self.write_tensorboard(logs, epoch)
        print(
            f'''Val Epoch: {epoch} [100%] || Time: {m} min {s} || P(v): {mean_pv:.3f} || P(v,h): {mean_pvh:.3f} || Acc: {mean_acc:.3f}''')
