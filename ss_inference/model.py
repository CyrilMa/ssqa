import time

import torch
from torch import nn
from torch.nn import functional as F


from .layers import ResBlock, ConvBlock

class BaseNet(nn.Module):
    def __init__(self, in_channels):
        super(BaseNet, self).__init__()
        self.in_channels = in_channels
        self.device = "cpu"
        
    def to(self, device):
        super(BaseNet, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        pass

    def train_epoch(self, loader, optimizer, epoch = 0, verbose = 2):
        n_res, mean_ss3, mean_ss8, mean_box, mean_other, mean_loss, mean_ss3_acc, mean_ss8_acc = 0, 0, 0, 0, 0, 0, 0, 0
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
                print(f'''Train Epoch: {epoch} [{int(100 * batch_idx / len(loader))}%] || Time: {m} min {s} ||\
                 SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc : {mean_ss8_acc:.3f} || Loss: {mean_loss:.3f} || \
                 SS3 Loss: {mean_ss3:.3f} || SS8 Loss: {mean_ss8:.3f} || Other Loss: {mean_other:.3f}'''
                    , end="\r")
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        if verbose > 0:
            print(f'''Train Epoch: {epoch} [100%] || Time: {m} min {s}  || SS3 Acc: {mean_ss3_acc:.3f} || SS8 Acc \
            : {mean_ss8_acc:.3f} || Loss: {mean_loss:.3f} || SS3 Loss: {mean_ss3:.3f} || SS8 Loss: {mean_ss8:.3f} \
            || Other Loss: {mean_other:.3f}''')
        return mean_ss3_acc, mean_ss8_acc, mean_loss, mean_ss3, mean_ss8

    def val_epoch(self, loader, epoch = 0, verbose = 2):
        mean_ss3_acc, mean_ss8_acc, n_res = 0, 0, 0
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
        m, s = int(time.time() - start) // 60, int(time.time() - start) % 60
        if verbose > 0:
            print(
                f'''Val Epoch: {epoch} [100%] || Time: {m} min {s} || SS3 Acc: {mean_ss3_acc:.3f} || \
                SS8 Acc: {mean_ss8_acc:.3f}''')
        return mean_ss3_acc, mean_ss8_acc

    def predict(self, loader):
        self.eval()
        ss3, ss8, others = [], [],[]
        for batch_idx, (x,t, is_empty) in enumerate(loader):
            x = x.float().permute(0, 2, 1).to(self.device)[:, :50]
            is_empty = is_empty.float().permute(0, 2, 1).to(self.device)

            B, _, N = x.size()

            p_other, p_ss8, p_ss3 = self(x, is_empty)
            p_other, p_ss8, p_ss3 = p_other.cpu(), p_ss8.cpu(), p_ss3.cpu()

            p_ss3 = F.softmax(p_ss3, 1)
            p_ss8 = F.softmax(p_ss8, 1)
            ss3.append(p_ss3[0])
            ss8.append(p_ss8[0])
            others.append(p_other[0])
        return others, ss8, ss3

class NetSurfP2(BaseNet):
    def __init__(self, in_channels):
        super(NetSurfP2, self).__init__(in_channels)
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

    def __init__(self, in_channels):
        super(ConvNet, self).__init__(in_channels)
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

    def forward(self, x):
        B, _, N = x.size()
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


class HmmConvNet(BaseNet):
    def __init__(self, in_channels):
        super(HmmConvNet, self).__init__(in_channels)
        self.conv1 = ResBlock(in_channels, 100, 11)
        self.conv2 = ResBlock(100, 200, 11)
        self.conv3 = ResBlock(200, 400, 11)
        self.conv4 = ResBlock(400, 200, 11)
        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                                  200, 3, 1,
                                  stride=1, padding=0, dilation=1)

    def forward(self, x):
        B, _, N = x.size()
        is_empty = (x.max(1).values != 0).int().view(B, 1, N)

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h_ss3 = self.conv_ss3(h) * is_empty
        return None, None, h_ss3


