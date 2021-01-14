import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

log21 = np.log(21)


class MaximumEntropyGenerator(nn.Module):

    def __init__(self, E, G, H, log_dir, train_E=False):
        super(MaximumEntropyGenerator, self).__init__()
        self.E = E
        self.G = G
        self.H = H
        self.log_dir = log_dir
        self.gen_writer = SummaryWriter(self.log_dir)

        self.z_dim = G.z_dim
        self.device = "cpu"

        self.optimizerG = optim.Adam(G.parameters(), lr=0.001)
        self.G.train()

        if E is not None:
            self.optimizerE = optim.Adam(E.parameters(), lr=0.001)
            if train_E:
                self.E.train()

        if H is not None:
            self.optimizerH = optim.Adam(H.parameters(), lr=0.001)
            self.H.train()

        self.g_hist = []
        self.e_hist = []

    def train_generator(self, batch_size):
        self.G.zero_grad()
        self.H.zero_grad()

        z = torch.randn(batch_size, self.z_dim).to(self.device)
        x = self.G(z)
        D_fake, seq, struct = self.E(x)
        D_fake = D_fake.mean()

        y = torch.zeros(2 * batch_size).to(self.device)
        y[:batch_size].data.fill_(1)

        z_bar = z[torch.randperm(batch_size)]
        concat_x = torch.cat([x, x], 0)
        concat_z = torch.cat([z, z_bar], 0)
        mi_estimate = nn.BCEWithLogitsLoss()(self.H(concat_x, concat_z).squeeze(), y)

        (D_fake + 5 * mi_estimate).backward()

        self.optimizerG.step()
        self.optimizerH.step()
        logs = [D_fake.item(), mi_estimate.item(), seq.item(), struct.item(), len(self.g_hist)]
        self.g_hist.append(logs)
        self.write_tensorboard_gen(logs)
        return logs

    def train_energy_model(self, x_real):
        batch_size = x_real.size(0)
        self.E.zero_grad()

        d_real = {"sequence": x_real}
        D_real = self.E(d_real)
        D_real = D_real.mean()

        # train with fake
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        x_fake = self.G(z).detach()
        d_fake = {"sequence": x_fake}
        D_fake = self.E(d_fake)
        D_fake = D_fake.mean()

        penalty = self.E.energy(x_real)
        (D_real - D_fake).backward()

        self.optimizerE.step()

        self.e_hist.append([D_real.item(), D_fake.item(), None])
        return self.e_hist

    def write_tensorboard_gen(self, logs):
        D_fake, mi_estimate, seq, struct, n_iter = logs
        self.gen_writer.add_scalar('E(DCA)', seq, n_iter)
        self.gen_writer.add_scalar('E(ssqa)', struct, n_iter)
        self.gen_writer.add_scalar('I(JSD)', mi_estimate, n_iter)
        self.gen_writer.add_scalar('Loss', D_fake, n_iter)

    def generate(self, batch_size, seed=None):
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        if seed is not None:
            z = seed
        return self.G(z).detach()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def freeze_E(self):
        self.E.eval()

    def freeze_G(self):
        self.G.eval()

    def freeze_H(self):
        self.H.eval()

    def train_E(self):
        self.E.train()

    def train_G(self):
        self.G.train()

    def train_H(self):
        self.H.train()