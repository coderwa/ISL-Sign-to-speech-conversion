"""Compact residual MLP for static ISL handshape classification.

A static sign is a single pose, so a recurrent model over 16 *copies* of the
same frame adds latency and parameters without adding information.
"""
import torch
from torch import nn
from config import INPUT_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ISLLandmarkNet(nn.Module):
    def __init__(self, num_classes=35, input_size=INPUT_SIZE, hidden_size=256, num_blocks=3, dropout=0.25):
        super().__init__()
        self.input_size = input_size
        self.stem = nn.Sequential(nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input (batch, {self.input_size}), received {tuple(x.shape)}")
        return self.head(self.blocks(self.stem(x)))


def build_model(num_classes=35, **kwargs):
    return ISLLandmarkNet(num_classes=num_classes, **kwargs)
