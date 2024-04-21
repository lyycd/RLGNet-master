import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv

import config
from conv import RelGraphConv, RGCNBlockLayer


class RGCN(nn.Module):
    def __init__(self, h_dim, num_r, num_bases=10, layers=1, dropout=0.2, layer_norm=True, self_loop=True, bias=True, activation=None):
        super(RGCN, self).__init__()
        self.RGCN_layers = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            self.RGCN_layers.append(
                RelGraphConv(in_feat=h_dim, out_feat=h_dim, num_rels=num_r, regularizer='bdd', num_bases=num_bases, dropout=dropout,
                             layer_norm=layer_norm, self_loop=self_loop, bias=bias, activation=activation))

    def message_func(self, edges):
        return {'m': edges.src['h']}

    def forward(self, g, h, rel):
        out = [h]
        for i in range(self.layers):
            res = self.RGCN_layers[i](g, out[-1], rel)
            out.append(res)
        return out[-1]


class RelGCN(nn.Module):
    def __init__(self, h_dim, num_r, num_bases=-1, layers=1, dropout=0.2, layer_norm=True, self_loop=True, bias=True, activation=None):
        super(RelGCN, self).__init__()
        self.h_dim = h_dim
        self.RGCN_layers = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            self.RGCN_layers.append(
                RGCNBlockLayer(in_feat=h_dim, out_feat=h_dim, num_rels=num_r, num_bases=num_bases, dropout=dropout, layer_norm=layer_norm,
                               self_loop=False, bias=False, activation=activation))

    def forward(self, g, h):
        out = [h]
        for i in range(self.layers):
            res = self.RGCN_layers[i](g, out[-1], g.edata[dgl.ETYPE])
            out.append(res)

        return out[-1]
