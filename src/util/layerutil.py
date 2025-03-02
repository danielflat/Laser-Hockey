import torch
from torch import nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, simnorm_dim = 8):
        shape = x.shape
        x = x.view(*shape[:-1], -1, simnorm_dim)
        x = F.softmax(x, dim = -1)
        return x.view(*shape)


class NormedLinear(nn.Module):
    """
    Step 01: Linear Layer
    Step 02 (Optional): Dropout
    Step 03: LayerNorm
    Step 04: Activation Function (Mish and SimNorm are supported)
    """

    def __init__(self, in_features: int, out_features: int, activation_function: str, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.dropout = nn.Dropout(dropout, inplace = False) if dropout else None
        self.layer_norm = nn.LayerNorm(out_features)

        if activation_function == "Mish":
            self.activation_function = nn.Mish(inplace = False)
        elif activation_function == "SimNorm":
            self.activation_function = SimNorm()
        else:
            raise NotImplementedError(f"Activation function {activation_function} not implemented.")

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.activation_function(x)
        return x
