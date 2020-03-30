import torch
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

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvNet, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = ConvBlock(nn.Conv1d, nn.PReLU, nn.BatchNorm1d, 
                                in_channels, 100, 11,
                                stride=1, padding=5, dilation=1)
        self.conv2 = ConvBlock(nn.Conv1d, nn.PReLU, nn.BatchNorm1d, 
                                100, 100, 11,
                                stride=1, padding=5, dilation=1)
        self.conv3 = ConvBlock(nn.Conv1d, nn.PReLU, nn.BatchNorm1d, 
                                100, 100, 11,
                                stride=1, padding=5, dilation=1)
        self.conv4 = ConvBlock(nn.Conv1d, nn.PReLU, nn.BatchNorm1d, 
                                100, 100, 11,
                                stride=1, padding=5, dilation=1)
        self.conv4 = ConvBlock(nn.Conv1d, nn.PReLU, nn.BatchNorm1d, 
                        100, 100, 11,
                        stride=1, padding=5, dilation=1)
        
        self.conv5 = ConvBlock(nn.Conv1d,nn.PReLU, nn.BatchNorm1d,
                        100, 100, 11,
                        stride=1, padding=5, dilation=1)
        self.linear2 = ConvBlock(nn.Conv1d, None, None, 
                100, out_channels, 1,
                stride=1, padding=0, dilation=1)



    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
#         h1 = self.linear1(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
#         h = torch.cat([h, h1], 1)
        return self.linear2(h)