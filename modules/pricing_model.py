import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
from bisect import bisect
from pathlib import Path

from modules.tcn import TemporalConvNet, TemporalBlock
from modules.yield_curve import YieldCurve


class PricingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
                
        self.tcn = TemporalConvNet(
            num_inputs=self.config.tcn_input_size, 
            num_channels=self.config.tcn_num_channels, 
            kernel_size=self.config.tcn_kernel_size, 
            dropout=self.config.tcn_dropout
        )
        self.tcn_pooling = TemporalBlock(
            n_inputs=self.config.tcn_num_channels[-1],
            n_outputs=1, 
            kernel_size=self.config.tcn_kernel_size, 
            stride=1, 
            dilation=2**(len(self.config.tcn_num_channels) + 1),
            padding=(self.config.tcn_kernel_size-1) * 2**(len(self.config.tcn_num_channels) + 1), 
            dropout=self.config.tcn_dropout
        )
        self.layer_norm = nn.LayerNorm(self.config.seq_len)
        self.linears = Linears(
            input_size=self.config.seq_len + self.config.tabular_data_size, 
            hidden_sizes=self.config.linear_sizes, 
            output_size=self.config.n_targets, 
            dropout=self.config.linear_dropout
        )
        # self.activation_fn = nn.ReLU()
        self.activation_fn = nn.Sigmoid()

    def forward(self, x):
        option_data, ts = x
        # ts needs to have dimension (N, C, L) in order to be passed into TCN
        tcn_output = self.tcn(ts)
        pooled_tcn_output = torch.squeeze(self.tcn_pooling(tcn_output))
        normalized_output = self.layer_norm(pooled_tcn_output)
        x = torch.cat((option_data, normalized_output), axis=1)
        for linear in self.linears:
            x = linear(x)
        x = self.activation_fn(torch.squeeze(x))
        return x


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