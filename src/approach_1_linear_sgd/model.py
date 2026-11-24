"""A single affine layer with continuous price output in billion VND."""

from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)
