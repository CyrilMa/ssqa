import time
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from .layers import ResBlock, ConvBlock

DATA = '/home/malbranke/data'

class BaseNet(nn.Module):
    r"""
    Base class for Secondary Structure Inference Network

    Args:
        in_channels (int): number of input channels
        name (str): name of the network (change for each experiment)
    """

    def __init__(self, in_channels, name):
        super(BaseNet, self).__init__()
        self.in_channels = in_channels
        self.device = "cpu"
        self.name = name
        self.writer = SummaryWriter(f"{DATA}/tensorboard/ss_inf/{self.name}-{self.in_channels}")

    def __repr__(self):
        return f"Model {self.name}-{self.in_channels}"

    def to(self, device):
        super(BaseNet, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        pass

    def write_tensorboard(self, logs, n_iter):
        r"""

        Args:
            logs (dict): data to write in Tensorboard, name of the metric in key, value in value
            n_iter (int): number of te iteration to write Z
        """
        for k, v in logs.items():
            self.writer.add_scalar(k, v, n_iter)

    def train_epoch(self, loader, optimizer, epoch = 0, verbose = 2):
        r"""
        Train through one epoch

        Args:
            loader (DataLoader): Loader for training data
            optimizer (Optimizer): Optimizer to use
            epoch (int): number of the epoch
            verbose (int): 0 = silence | 1 = One print per iteration | 2 = One return per iteration

        """
        n_res, mean_ss3, mean_ss8, mean_box, mean_other, mean_loss, mean_ss3_acc, mean_ss8_acc = 0., 0., 0., 0., 0., 0., 0., 0.
        start = time.time()
        self.train()
        for batch_idx, (x, t, is_empty) in enumerate(loader):
            x = x.float().permute(0, 2, 1).to(self.device)
            t = t.float().permute(0, 2, 1).to(self.device)
            is_empty = is_empty.float().permute(0, 2, 1).to(self.device)
            B, _, N = x.size()

            optimizer.zero_grad()
            p_other, p_ss8, p_ss3 = self(x, is_empty)
            p_other, p_ss8, p_ss3 = p_other.cpu(), p_ss8.cpu(), p_ss3.cpu()
            x, t, is_empty = x.cpu(), t.cpu(), is_empty.cpu()

            ss3_acc = ((p_ss3.argmax(1) == t[:, -1]) * is_empty[:, 0]).int().sum() / is_empty[:, 0].sum()
            ss8_acc = ((p_ss8.argmax(1) == t[:, -2]) * is_empty[:, 0]).int().sum() / is_empty[:, 0].sum()
            ss3_loss = F.cross_entropy(p_ss3, t[:, -1].long())
            ss8_loss = F.cross_entropy(p_ss8, t[:, -2].long())
            other_loss = F.mse_loss(p_other, t[:, :9]) / 1500

            loss = ss3_loss + ss8_loss + other_loss
            loss.backward()
            optimizer.step()

            del x
            mean_ss3 = (mean_ss3 * batch_idx + ss3_loss.item()) / (batch_idx + 1)
            mean_ss8 = (mean_ss8 * batch_idx + ss8_loss.item()) / (batch_idx + 1)
            mean_other = (mean_other * batch_idx + other_loss.item()) / (batch_idx + 1)
            mean_loss = (mean_loss * batch_idx + loss.item()) / (batch_idx + 1)
            mean_ss3_acc = (mean_ss3_acc * n_res + N * ss3_acc.item()) / (n_res + N)
            mean_ss8_acc = (mean_ss8_acc * n_res + N * ss8_acc.item()) / (n_res + N)
            n_res += N
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
            if verbose > 1:
                print(f'''Train Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} || SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc : {mean_ss8_acc:.3f} || Loss: {mean_loss:.3f} || SS3 Loss: {mean_ss3:.3f} || SS8 Loss: {mean_ss8:.3f} || Other Loss: {mean_other:.3f}'''
                    , end="\r")
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        logs = {"train/ss3_acc": mean_ss3_acc, "train/ss8_acc" : mean_ss8_acc, "train/loss": mean_loss,
                "train/ss3_loss": mean_ss3, "train/ss8_loss": mean_ss8, "train/other": mean_other}
        self.write_tensorboard(logs, epoch)
        if verbose > 0:
            print(f'''Train Epoch: {epoch} [100%] || Time: {m} min {s}  || SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc : {mean_ss8_acc:.3f} || Loss: {mean_loss:.3f} || SS3 Loss: {mean_ss3:.3f} || SS8 Loss: {mean_ss8:.3f} || Other Loss: {mean_other:.3f}''')
        return mean_ss3_acc, mean_ss8_acc, mean_loss, mean_ss3, mean_ss8

    def val_epoch(self, loader, epoch = 0, verbose = 2):
        r"""
        Validation through one epoch

        Args:
            loader (DataLoader): Loader for training data
            epoch (int): number of the epoch
            verbose (int): 0 = silence | 1 = One print per iteration | 2 = One return per iteration

        """

        mean_ss3_acc, mean_ss8_acc, n_res = 0., 0., 0.
        start = time.time()
        self.eval()
        for batch_idx, (x, t, is_empty) in enumerate(loader):
            x = x.float().permute(0, 2, 1).to(self.device)
            t = t.float().permute(0, 2, 1).to(self.device)
            is_empty = is_empty.float().permute(0, 2, 1).to(self.device)
            B, _, N = x.size()

            p_other, p_ss8, p_ss3 = self(x, is_empty)
            p_other, p_ss8, p_ss3 = p_other.cpu(), p_ss8.cpu(), p_ss3.cpu()
            x, t, is_empty = x.cpu(), t.cpu(), is_empty.cpu()

            p_ss3 = F.softmax(p_ss3, 1)
            p = p_ss3.detach()

            n = is_empty[:, 0].sum()
            ss3_acc = ((p.argmax(1) == t[:, -1]) * is_empty[:, 0]).int().sum() / n
            ss8_acc = ((p_ss8.argmax(1) == t[:, -2]) * is_empty[:, 0]).int().sum() / n
            mean_ss3_acc = (mean_ss3_acc * n_res + n * ss3_acc.item()) / (n_res + n)
            mean_ss8_acc = (mean_ss8_acc * n_res + n * ss8_acc.item()) / (n_res + n)
            n_res += n
            m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
            if verbose > 1:
                print(
                    f'''Val Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} || \
                    SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc: {mean_ss8_acc:.3f}''',
                    end="\r")

        logs = {"val/ss3_acc": mean_ss3_acc, "val/ss8_acc" : mean_ss8_acc}
        self.write_tensorboard(logs, epoch)
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        if verbose > 0:
            print(
                f'''Val Epoch: {epoch} [100%] || Time: {m} min {s} || SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc: {mean_ss8_acc:.3f}''')
        return mean_ss3_acc, mean_ss8_acc

    def predict(self, loader):
        r"""
        Predict all data in a Data Loader

        Args:
            loader (DataLoader): Loader for training data

        """

        self.eval()
        ss3, ss8, others = [], [],[]
        for batch_idx, (x,_, is_empty) in tqdm(enumerate(loader)):
            x = x.float().permute(0, 2, 1).to(self.device)[:, :50]
            is_empty = is_empty.float().permute(0, 2, 1).to(self.device)

            B, _, N = x.size()

            p_other, p_ss8, p_ss3 = self(x, is_empty)
            p_other, p_ss8, p_ss3 = p_other.cpu(), p_ss8.cpu(), p_ss3.cpu()

            p_ss3 = F.softmax(p_ss3, 1)
            p_ss8 = F.softmax(p_ss8, 1)
            ss3.append(p_ss3[0].detach())
            ss8.append(p_ss8[0].detach())
            others.append(p_other[0].detach())
            torch.cuda.empty_cache()
        return others, ss8, ss3

class NetSurfP2(BaseNet):
    r"""
    Model strongly derived from the NetSurfP2 model below

    Klausen, Michael Schantz, et al. "NetSurfP‐2.0: Improved prediction of protein structural features by integrated
    deep learning." Proteins: Structure, Function, and Bioinformatics 87.6 (2019): 520-527.

    Args:
        in_channels (int): number of input channels
        name (str): name of the network (change for each experiment)
    """

    def __init__(self, in_channels, name):
        super(NetSurfP2, self).__init__(in_channels, name)
        self.conv1 = ResBlock(in_channels, 32, 129)
        self.conv2 = ResBlock(32, 32, 257)
        self.lstm1 = nn.LSTM(input_size=32 + in_channels,
                             hidden_size=1024,
                             num_layers=1,
                             bias=True,
                             bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=2048,
                             hidden_size=1024,
                             num_layers=1,
                             bias=True,
                             bidirectional=True)
        self.conv_other = ConvBlock(nn.Conv1d, None, None,
                                    2048, 9, 1,
                                    stride=1, padding=0, dilation=1)
        self.conv_ss8 = ConvBlock(nn.Conv1d, None, None,

                                  2048, 8, 1,
                                  stride=1, padding=0, dilation=1)
        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                                  2048, 3, 1,
                                  stride=1, padding=0, dilation=1)

    def forward(self, x, is_empty=None):
        r"""
        Args:
            x (torch.FloatTensor): Input, dim = [batch_size, n_channels, N]
            is_empty  (torch.FloatTensor): Recomputed if None, dim = [batch_size, 1, N]
        """
        B, _, N = x.size()
        if is_empty is None:
            is_empty = (x.max(1).values != 0).int().view(B, 1, N)
        h = self.conv1(x)
        h = self.conv2(h)
        h = torch.cat([x, h], 1)
        h = h.permute(2, 0, 1)
        h = self.lstm1(h)[0]
        h = self.lstm2(h)[0].permute(1, 2, 0)

        h_other = self.conv_other(h) * is_empty
        h_ss8 = self.conv_ss8(h) * is_empty
        h_ss3 = self.conv_ss3(h) * is_empty
        return h_other, h_ss8, h_ss3


class ConvNet(BaseNet):
    r"""
    Simple Light Convolutional model that reach 80% accuracy on SS3

    Args:
        in_channels (int): number of input channels
        name (str): name of the network (change for each experiment)
    """


    def __init__(self, in_channels, name):
        super(ConvNet, self).__init__(in_channels, name)
        self.conv1 = ResBlock(in_channels, 200, 11)
        self.conv2 = ResBlock(200, 200, 11)
        self.conv3 = ResBlock(400, 200, 17)
        self.conv4 = ResBlock(600, 100, 17)
        self.conv_other = ConvBlock(nn.Conv1d, None, None,
                                    100, 9, 1,
                                    stride=1, padding=0, dilation=1)
        self.conv_ss8 = ConvBlock(nn.Conv1d, None, None,
                                  100, 8, 1,
                                  stride=1, padding=0, dilation=1)

        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                                  100, 3, 1,
                                  stride=1, padding=0, dilation=1)

    def forward(self, x, is_empty=None):
        r"""
        Args:
            x (torch.FloatTensor): Input, dim = [batch_size, n_channels, N]
            is_empty  (torch.FloatTensor): Recomputed if None, dim = [batch_size, 1, N]
        """
        B, _, N = x.size()
        if is_empty is None:
            is_empty = (x.max(1).values != 0).int().view(B, 1, N)
        h_1 = h =self.conv1(x)
        h_2 = self.conv2(h)
        h = torch.cat([h_2, h_1], 1)
        h_3 = self.conv3(h)
        h = torch.cat([h_3, h_2, h_1], 1)
        h = self.conv4(h)
        h_other = self.conv_other(h) * is_empty
        h_ss8 = self.conv_ss8(h) * is_empty
        h_ss3 = self.conv_ss3(h) * is_empty
        return h_other, h_ss8, h_ss3



