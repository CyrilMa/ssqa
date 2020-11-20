from torch import nn

def leaky_relu():
    return nn.LeakyReLU(0.2, inplace=False)

class ConvBlock(nn.Module):
    r"""
    Basic Convolutional Block with activation and normalization.
    """

    def __init__(self, conv, activation=None, normalization=None, in_channels = 0, out_channels = 0, *args, **kwargs):
        r"""

        Args:
            conv (nn.Module): Convolution layer
            activation (function): Activation layer
            normalization (nn.Module): Normalization Layer
            in_channels (int): number of channels in input
            out_channels (int): number of channels in output
            *args: args for Conv Layer
            **kwargs: kwargs for Conv Layer
        """
        super(ConvBlock, self).__init__()
        self.conv = conv(in_channels, out_channels, *args, **kwargs)
        self.activation, self.normalization = None, None
        if activation is not None:
            self.activation = activation()
        if normalization is not None:
            self.normalization = normalization(out_channels)

    def forward(self, x):
        h = self.conv(x)
        if self.normalization is not None:
            h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        return h

class ResBlock(nn.Module):
    r"""
    Basic Residual Block.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        r"""

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the kernel for the convolution
            bias (bool): True if use a bias in convolution
        """
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
