from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid,
               "elu": nn.ELU, "prelu": nn.PReLU}


class MLP(nn.Module):
    """Flatten the image, then `len(hidden)` blocks of Linear -> activation -> Dropout, then a
    linear output layer. With hidden=() this is multinomial logistic regression (the linear
    baseline of Q2.1).

    Args:
        in_dim: number of input features (3 * 64 * 64 = 12288)
        num_classes: number of output classes
        hidden: sequence of hidden-layer widths, e.g. (512, 256)
        activation: key of ACTIVATIONS
        dropout: dropout probability after each hidden activation
    """

    def __init__(self, in_dim=3 * 64 * 64, num_classes=10, hidden=(), activation="relu", dropout=0.0):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for width in hidden:
            layers += [nn.Linear(d, width), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def first_layer(self):
        """The first nn.Linear (its weight rows are the images visualized in Q2.4)."""
        return self.net[1]


class CNN(nn.Module):
    """A small VGG-style CNN: `len(channels)` blocks of [conv 3x3 (padding 1) -> (BatchNorm) ->
    activation] x `convs_per_block`, each block followed by 2x2 max pooling, then global average
    pooling over the remaining positions, dropout, and a linear output layer.

    Locality and weight sharing come from the 3x3 convolutions (Q1.1 (a)); padding 1 keeps the
    convolutions translation equivariant away from the border (Q1.1 (b)); global average pooling
    makes the feature vector (approximately) translation invariant. Max pooling with stride 2 only
    keeps exact equivariance for shifts that are multiples of 2 ** len(channels).

    Args:
        in_channels: 3 for RGB
        num_classes: number of output classes
        channels: output channels of each block, e.g. (32, 64, 128, 256)
        convs_per_block: number of 3x3 convolutions per block
        batchnorm: BatchNorm2d after each convolution
        activation: key of ACTIVATIONS
        dropout: dropout probability before the output layer

    `features(x)` returns the pooled feature vector (batch, channels[-1]), e.g. to use the CNN as
    an image encoder in Part 3.
    """

    def __init__(self, in_channels=3, num_classes=10, channels=(32, 64, 128, 256),
                 convs_per_block=1, batchnorm=True, activation="relu", dropout=0.0):
        super().__init__()
        layers, c_in = [], in_channels
        for c_out in channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not batchnorm))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(ACTIVATIONS[activation]())
                c_in = c_out
            layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def features(self, x):
        return self.pool(self.conv(x)).flatten(1)

    def forward(self, x):
        return self.head(self.pool(self.conv(x)))
