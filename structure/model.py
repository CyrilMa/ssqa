import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

def leaky_relu(): 
    return nn.LeakyReLU(0.2, inplace = False)

class ConvBlock(nn.Module):
    def __init__(self, conv, activation = None, normalization = None, *args, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = conv(*args, **kwargs)
        self.activation, self.normalization = None, None
        if activation is not None:
            self.activation = activation()
        if normalization is not None:
            self.normalization = normalization(args[1])
        
    def forward(self, x):
        h = self.conv(x)
        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        return h

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, bias=True):
        super(ResBlock, self).__init__()
        pad = (kernel_size-1)//2
        self.conv_1 = ConvBlock(nn.Conv1d, leaky_relu, nn.BatchNorm1d, 
                                in_channels, out_channels, kernel_size,
                                stride=1, padding=pad, bias=bias)
        self.conv2 = ConvBlock(nn.Conv1d, None, nn.BatchNorm1d, 
                                in_channels, out_channels, 1,
                                stride=1, padding=0, bias=bias)

    def forward(self, x):
        identity = x
        out = self.conv_1(x)
        identity = self.conv2(x)
        out += identity
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels, heads = 1, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(channels, heads, *args, **kwargs)
    
    def forward(self, x):
        h = x.permute(2,0,1)
        h, _ = self.attention(h, h, h)
        h = h.permute(1,2,0)
        return h
    
class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels = 100, bias=True):
        super(ConvNet, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = ResBlock(in_channels, 100, 11)
        self.conv2 = ResBlock(100, 200, 11)
        self.conv3 = ResBlock(200, 400, 11)
        self.conv4 = ResBlock(400, 200, 11)
        self.conv_other = ConvBlock(nn.Conv1d, None, None,
                        200, 9, 1,
                        stride=1, padding=0, dilation=1)
        self.conv_ss8 = ConvBlock(nn.Conv1d, None, None,
                            200, 8, 1,
                            stride=1, padding=0, dilation=1)
        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                            200, 3, 1,
                            stride=1, padding=0, dilation=1)

    
    def forward(self, x):
        B, _, N = x.size()
        C = torch.tensor(range(N)).cuda().view(1, 1,-1)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h_other = self.conv_other(h)
        h_ss8 = self.conv_ss8(h)
        h_ss3 = self.conv_ss3(h)
        return h_other, h_ss8, h_ss3

class HmmConvNet(nn.Module):
    def __init__(self, in_channels, out_channels = 100, bias=True):
        super(HmmConvNet, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = ResBlock(in_channels, 100, 11)
        self.conv2 = ResBlock(100, 200, 11)
        self.conv3 = ResBlock(200, 400, 11)
        self.conv4 = ResBlock(400, 200, 11)
        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                            200, 3, 1,
                            stride=1, padding=0, dilation=1)

    
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h_ss3 = self.conv_ss3(h)
        return h_ss3

class LSTMNet(nn.Module):
    def __init__(self, in_channels, out_channels = 100, N = 128, bias=True):
        super(LSTMNet, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = ResBlock(in_channels, 32, 129)
        self.conv2 = ResBlock(32, 32, 257)
        self.lstm1 = nn.LSTM(input_size = 32, 
                            hidden_size = 512, 
                            num_layers = 2,
                            bias = True,
                            bidirectional = True)
        self.lstm2 = nn.LSTM(input_size = 1024, 
                    hidden_size = 512, 
                    num_layers = 2,
                    bias = True,
                    bidirectional = True)
        self.conv_other = ConvBlock(nn.Conv1d, None, None,
                        1024, 9, 1,
                        stride=1, padding=0, dilation=1)
        self.conv_ss8 = ConvBlock(nn.Conv1d, None, None,
                            
                                  1024, 8, 1,
                            stride=1, padding=0, dilation=1)
        self.conv_ss3 = ConvBlock(nn.Conv1d, None, None,
                            1024, 3, 1,
                            stride=1, padding=0, dilation=1)

        self.conv5 = ResBlock(1024, 100, 129)
        self.conv_box = ConvBlock(nn.Conv1d, nn.ReLU, None,
                            120, 2, 1,
                            stride=1, padding=0, dilation=1)

        
    def forward(self, x):
        B, _, N = x.size()
        C = torch.tensor(range(N)).cuda().view(1, 1,-1)
        h = (x-MEAN)/STD

        h = self.conv1(x)
        h = self.conv2(h)
        h = h.permute(2, 0, 1)
        h = self.lstm1(h)[0]
        h = self.lstm2(h)[0].permute(1, 2, 0)
        
        h_other = self.conv_other(h)
        h_ss8 = self.conv_ss8(h)
        h_ss3 = self.conv_ss3(h)
        h = self.conv5(h)
        h = torch.cat([h,h_other,h_ss8,h_ss3],1)
        h_box = self.conv_box(h)
        h_box = torch.cat([C-h_box[:,0:1], C+h_box[:,1:2]],1)
        return h_other, h_ss8, h_ss3, h_box
