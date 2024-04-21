import copy
import math

import dgl
import torch
import torch as th
from dgl.nn.pytorch import TypedLinear
import dgl.function as fn
from torch import nn
import torch.nn.functional as F
import config


class RelGraphConv(nn.Module):

    def __init__(
            self,
            in_feat,
            out_feat,
            num_rels,
            regularizer=None,
            num_bases=None,
            bias=True,
            activation=None,
            self_loop=True,
            dropout=0.0,
            layer_norm=False,
    ):

        super().__init__()
        self.num_r = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # TODO(minjie): consider remove those options in the future to make
        #   the module only about graph convolution.
        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(
                out_feat, elementwise_affine=True
            )

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        self.weight_neighbor = nn.Parameter(torch.Tensor(in_feat * 4, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        self.weight_neighbor_one = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor_one, gain=nn.init.calculate_gain('relu'))
        self.rel = None

        self.w1 = nn.Parameter(torch.Tensor(in_feat * 4, out_feat))
        nn.init.xavier_uniform_(self.w1, gain=nn.init.calculate_gain('relu'))

    def message(self, edges):
        """Message function."""
        e_type = edges.data[dgl.ETYPE]
        rel = self.rel[e_type]
        m1 = edges.src["h"] + rel
        m2 = edges.src["h"] * rel
        m = torch.cat([m1, m2, edges.src["h"], rel], dim=1)
        m = torch.mm(m, self.weight_neighbor)
        return {"m": m}

    def forward(self, g, feat, rel=None):
        if rel is not None:
            self.rel = rel
        with g.local_scope():
            g.srcdata["h"] = feat
            if self.self_loop:
                loop_message = torch.mm(g.ndata['h'], self.loop_weight)

            g.edata['etype'] = g.edata[dgl.ETYPE]
            g.update_all(self.message, fn.mean("m", "h"))
            h = g.dstdata["h"]

            if self.self_loop:
                h = h + loop_message
            if self.activation:
                h = self.activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            return h


class RGCNBlockLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None, activation=None, self_loop=False, dropout=0.0, layer_norm=False):
        super().__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.self_loop = self_loop
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def message(self, edges):
        weight = self.weight.index_select(0, edges.data['etype']).view(
            -1, self.submat_in, self.submat_out)  # [edge_num, submat_in, submat_out]
        node = edges.src['h'].view(-1, 1, self.submat_in)  # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)  # [edge_num, out_feat]
        return {'m': msg}

    def forward(self, g, feat, rel_type):

        with g.local_scope():
            g.srcdata["h"] = feat
            if self.self_loop:
                loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            g.edata['etype'] = g.edata[dgl.ETYPE]
            g.update_all(self.message, fn.mean("m", "h"))
            h = g.dstdata["h"]
            if self.self_loop:
                h = h + loop_message
            if self.activation:
                h = self.activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
            return h
