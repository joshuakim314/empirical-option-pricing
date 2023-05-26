import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm

import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
from bisect import bisect
from pathlib import Path

from modules.tcn import TemporalConvNet
from modules.yield_curve import YieldCurve


class PricingModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, seq_len, data_len, linear_sizes, kernel_size, tcn_dropout, linear_dropout):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=tcn_dropout)
        self.linear = Linears(seq_len + data_len, linear_sizes, output_size, dropout=linear_dropout)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        option_data, ts = x
        # ts needs to have dimension (N, C, L) in order to be passed into TCN
        tcn_output = torch.squeeze(self.tcn(ts).transpose(1, 2))
        x = torch.cat((option_data, tcn_output), axis=1)
        for linear in self.linears:
            x = linear(x)
        return self.activation_fn(torch.squeeze(x))


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Linears(input_size, hidden_sizes, output_size, dropout=None):
    # builds block of linear layers; applies batch norm if dropout is none
    linears = nn.ModuleList()
    for i, (in_sz, out_sz) in enumerate(zip([input_size] + hidden_sizes, hidden_sizes + [output_size])):
        if i != len(hidden_sizes):
            linears.append(
                nn.Sequential(
                    Linear(in_sz, out_sz),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout else nn.BatchNorm1d(out_sz)
                )
            )
        else:
            linears.append(
                nn.Sequential(
                    Linear(in_sz, out_sz)
                )
            )
    return linears


if __name__ == '__main__':
    pass