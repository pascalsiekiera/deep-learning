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
