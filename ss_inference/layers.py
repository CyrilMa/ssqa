from torch import nn

def leaky_relu():
    return nn.LeakyReLU(0.2, inplace=False)

class ConvBlock(nn.Module):
    def __init__(self, conv, activation=None, normalization=None, *args, **kwargs):
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
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super(ResBlock, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv_1 = ConvBlock(nn.Conv1d, leaky_relu, nn.BatchNorm1d,
                                in_channels, out_channels, kernel_size,
                                stride=1, padding=pad, bias=bias)
        self.conv2 = ConvBlock(nn.Conv1d, None, nn.BatchNorm1d,
                               in_channels, out_channels, 1,
                               stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.conv_1(x)
        identity = self.conv2(x)
        out += identity
        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, heads=1, *args, **kwargs):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(channels, heads, *args, **kwargs)

    def forward(self, x):
        h = x.permute(2, 0, 1)
        h, _ = self.attention(h, h, h)
        h = h.permute(1, 2, 0)
        return h