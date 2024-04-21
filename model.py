import numpy as np
import torch

import config
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder, Decoder2, DecoderT

from rgcn import RGCN, RelGCN
import dgl.function as fn


class EntEmb(nn.Module):
    def __init__(self, use_static_graph, num_e, init='xavier', h_dim=-1, self_loop=True, activation=None):
        super(EntEmb, self).__init__()
        self.num_e = num_e
        self.activation = activation
        self.self_loop = self_loop
        if h_dim == -1:
            self.h_dim = config.h_dim
        else:
            self.h_dim = h_dim
        self.use_static_graph = use_static_graph
        self.loop_weight = nn.Parameter(torch.Tensor(self.h_dim, self.h_dim))
        nn.init.xavier_uniform_(
            self.loop_weight, gain=nn.init.calculate_gain("relu")
        )

        if use_static_graph is True:

            self.static_graph = config.static_graph
            self.static_graph = self.static_graph.to(config.device)
            self.static_num_e = config.static_num_e
            self.static_num_r = config.static_num_r
            self.word_ent = nn.Parameter(torch.zeros(self.static_num_e + self.num_e, self.h_dim))
            self.rgcn = RelGCN(h_dim=self.h_dim, num_r=self.static_num_r * 2, layers=1, num_bases=100, layer_norm=False)
            if init == 'xavier':
                nn.init.xavier_uniform_(self.word_ent, gain=nn.init.calculate_gain('relu'))
            else:
                torch.nn.init.normal_(self.word_ent)
        else:
            self.ent = nn.Parameter(torch.zeros(self.num_e, self.h_dim))
            if init == 'xavier':
                nn.init.xavier_uniform_(self.ent, gain=nn.init.calculate_gain('relu'))
            else:
                torch.nn.init.normal_(self.ent)
        self.dropout = nn.Dropout(0.2)

    def forward(self):
        if self.use_static_graph:
            out = self.rgcn(self.static_graph, F.normalize(self.word_ent))[:self.num_e]
            if self.activation is not None:
                out = self.activation(out)
            return out
        else:
            out = self.ent
            if self.self_loop is False:
                return out
            return self.dropout(out) @ self.loop_weight


class CosTimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(CosTimeEncode, self).__init__()
        time_dim = expand_dim
        basis_freq = (1 / 10 ** np.linspace(start=0, stop=9, num=time_dim))
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(basis_freq).float()))
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts, dim=1):
        map_ts = ts.unsqueeze(dim=dim).to(config.device) * self.basis_freq + self.phase
        harmonic = torch.cos(map_ts)
        return harmonic


class AbsTimeEncode(torch.nn.Module):
    def __init__(self, h_dim, v=0, bias=True, act=None):
        super(AbsTimeEncode, self).__init__()
        if v == 0:
            basis_freq = (1 / 10 ** np.linspace(start=0, stop=9, num=h_dim))
            self.tim_init_embeds = torch.nn.Parameter((torch.from_numpy(basis_freq).float()))
            self.phase = torch.nn.Parameter(torch.zeros(h_dim).float())
        else:
            self.tim_init_embeds = nn.Parameter(torch.zeros(1, h_dim))
            self.phase = torch.nn.Parameter(torch.zeros(h_dim).float())
            nn.init.xavier_uniform_(self.tim_init_embeds, gain=nn.init.calculate_gain('relu'))
        self.bias = bias
        self.tanh = nn.Tanh()
        self.act = act

    def forward(self, tim, dim=1):
        if self.bias:
            T = (tim.to(config.device) + 1).unsqueeze(dim=dim) * self.tim_init_embeds + self.phase
        else:
            T = (tim.to(config.device) + 1).unsqueeze(dim=dim) * self.tim_init_embeds
        if self.act is not None:
            T = self.act(T)
        return T


class UnionTimeEncode(torch.nn.Module):
    def __init__(self, h_dim, act=None):
        super(UnionTimeEncode, self).__init__()
        self.dim = h_dim // 2
        self.tim1 = AbsTimeEncode(self.dim, act=act)
        self.tim2 = CosTimeEncode(self.dim)

    def forward(self, t, dim=1):
        return torch.cat([self.tim1(t, dim=dim).squeeze(dim=1), self.tim2(t, dim=dim).squeeze(dim=1)], dim=dim)


class GlobalEncoder(nn.Module):
    def __init__(self, h_dim):
        super(GlobalEncoder, self).__init__()
        self.h_dim = h_dim
        self.t_dim = config.h_dim
        # self.t_dim = 48
        self.num_e = config.num_e
        self.num_r = config.num_r * 2

        self.w2 = nn.Linear(self.h_dim * 2, self.h_dim)
        nn.init.xavier_uniform_(self.w2.weight, gain=nn.init.calculate_gain('relu'))
        self.w1 = nn.Linear(self.h_dim + self.t_dim, self.h_dim)
        nn.init.xavier_uniform_(self.w1.weight, gain=nn.init.calculate_gain('relu'))
        self.w4 = nn.Linear(self.h_dim * 2, self.h_dim)
        nn.init.xavier_uniform_(self.w4.weight, gain=nn.init.calculate_gain('relu'))
        self.w3 = nn.Linear(self.h_dim + self.t_dim, self.h_dim)
        nn.init.xavier_uniform_(self.w3.weight, gain=nn.init.calculate_gain('relu'))

        self.ent = EntEmb(use_static_graph=config.use_static_graph, num_e=self.num_e)
        self.rel = EntEmb(use_static_graph=False, self_loop=False, num_e=self.num_r)

        self.pad_vec1 = nn.Parameter(torch.zeros(1, self.h_dim))
        self.pad_vec2 = nn.Parameter(torch.zeros(1, self.h_dim))
        nn.init.xavier_uniform_(self.pad_vec1, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.pad_vec2, gain=nn.init.calculate_gain('relu'))

        self.union_cnt = UnionTimeEncode(h_dim=self.h_dim, act=F.tanh)
        self.union_tim2 = UnionTimeEncode(h_dim=h_dim, act=F.tanh)

        self.decoder = Decoder2(0.2)

    def forward(self, data, cut_t, cnt, neg_t, neg_cnt):
        t = data[:, 3].to(config.device)
        ent = self.ent()
        ent = F.normalize(ent)
        rel = self.rel()
        s = data[:, 0]
        r = data[:, 1]
        s_emb = ent[s]
        r_emb = rel[r]
        ent_tmp1 = torch.cat([ent, self.pad_vec1])
        ent_tmp2 = torch.cat([ent, self.pad_vec2])
        neg_t_emb = ent_tmp1[neg_t]
        neg_cnt_emb = ent_tmp2[neg_cnt]

        cnt_emb = self.union_cnt(cnt, dim=2)
        s_r = torch.cat([s_emb, r_emb], dim=1)
        neg_emb = torch.cat([cnt_emb, neg_cnt_emb], dim=2)
        att2 = torch.bmm(F.relu(self.w4(s_r).unsqueeze(dim=1)),
                         F.relu(self.w3(neg_emb)).transpose(1, 2)).squeeze(dim=1)
        att2 = torch.softmax(att2, dim=1)
        cut_t_emb = self.union_tim2(cut_t, dim=2)
        s_r = torch.cat([s_emb, r_emb], dim=1)
        neg_emb = torch.cat([cut_t_emb, neg_t_emb], dim=2)
        att = torch.bmm(F.relu(self.w2(s_r).unsqueeze(dim=1)),
                        F.relu(self.w1(neg_emb)).transpose(1, 2)).squeeze(dim=1)
        att = torch.softmax(att, dim=1)
        g_tim_emb = self.union_tim2(t)
        neg_t_emb = neg_t_emb * att.unsqueeze(dim=2)
        neg_t_emb = neg_t_emb.sum(dim=1)

        neg_cnt_emb = neg_cnt_emb * att2.unsqueeze(dim=2)
        neg_cnt_emb = neg_cnt_emb.sum(dim=1)

        out = self.decoder(s, r, ent, rel, [neg_cnt_emb, neg_t_emb, g_tim_emb])

        return out


class LocalEncoder(nn.Module):
    def __init__(self, h_dim, num_layers):
        super(LocalEncoder, self).__init__()
        self.h_dim = h_dim
        self.num_e = config.num_e
        self.num_r = config.num_r
        self.num_layers = num_layers
        # self.t_dim = 200
        # self.t_dim = 128
        # self.t_dim = 96
        # self.t_dim = 64
        self.t_dim = 48
        # self.t_dim = 36

        self.union_time_encode = UnionTimeEncode(self.t_dim, act=F.tanh)

        self.rel = EntEmb(use_static_graph=False, num_e=self.num_r * 2, self_loop=False, init='normal', h_dim=self.h_dim)
        self.ent = EntEmb(use_static_graph=config.use_static_graph, num_e=self.num_e, self_loop=True, init='normal', h_dim=self.h_dim)

        self.rgcn = RGCN(h_dim=self.h_dim, num_r=self.num_r, num_bases=100, layers=config.num_layers, self_loop=True)

        self.W_b = nn.Linear(self.h_dim * 3 + self.t_dim, self.h_dim)
        nn.init.xavier_uniform_(self.W_b.weight, gain=nn.init.calculate_gain('relu'))
        self.W_c = nn.Linear(self.h_dim, 1)
        nn.init.xavier_uniform_(self.W_c.weight, gain=nn.init.calculate_gain('relu'))

        self.gru = nn.GRUCell(self.h_dim + self.t_dim, self.h_dim)
        self.decoder = Decoder(h_dim=self.h_dim, drop=0.2)

    def freeze(self):
        self.rel.requires_grad_(False)
        self.ent.requires_grad_(False)
        self.rgcn.requires_grad_(False)

    def message(self, edges):
        return {"m": edges.src["h"]}

    def forward(self, his_list_g, data, list_his):
        s, r, o, t = data.T
        s = s.clone().detach().to(config.device)
        batch = len(data)
        ent = self.ent()
        rel = self.rel()

        g_list = [F.normalize(ent)]
        rel = F.normalize(rel)

        tim_cnt = len(his_list_g)
        for g in his_list_g:
            g = g.to(config.device)

            h = g_list[-1]

            g.ndata['h'] = h
            g.update_all(self.message, fn.mean("m", "h"))
            h = g.ndata['h']
            h = F.normalize(h)

            tim_cnt -= 1
            in_degrees = g.in_degrees()
            zero_in_degree_nodes = torch.nonzero(in_degrees == 0, as_tuple=True)[0]

            tim = torch.ones([self.num_e, 1]).float() * tim_cnt
            tim[zero_in_degree_nodes] = 100
            cos_tim_emb = self.union_time_encode(tim)

            h = self.rgcn(g, h, rel)
            h = F.normalize(h)

            h = self.gru(torch.cat([h, cos_tim_emb], dim=1), g_list[-1])
            g_list.append(F.normalize(h))

        out = g_list[-1]

        tim_cnt = min(len(his_list_g), config.his_k)
        s_emb = []  #
        att2 = []
        for k in range(min(len(his_list_g), config.his_k)):
            tim_cnt -= 1
            lengths = torch.tensor([list_his[i][k].size(0) for i in range(batch)])
            y = torch.cat([list_his[i][k] for i in range(batch)])
            x = torch.cat([torch.tensor([i] * lengths[i]) for i in range(len(lengths))])
            sparse_tensor = torch.sparse_coo_tensor(torch.stack([x, y]).long(), torch.ones_like(x), (batch, self.num_e)).float().to(config.device)
            tmp = torch.sparse.mm(sparse_tensor, out)
            lengths[lengths == 0] = 1
            tmp = tmp / lengths.unsqueeze(dim=1).to(config.device)
            tmp = F.normalize(tmp)

            tim = torch.tensor([tim_cnt if length > 0 else 100 for length in lengths]).unsqueeze(dim=1)
            cos_tim_emb = self.union_time_encode(tim)

            s_emb.append(tmp)
            att2.append(self.W_c(torch.relu(self.W_b(torch.cat([out[s], rel[r], tmp, cos_tim_emb], dim=1)))))

        att2 = torch.stack(att2, dim=1)
        att2 = F.softmax(att2, dim=1)
        out2 = torch.stack(s_emb, dim=1)
        out2 = torch.bmm(out2.permute(0, 2, 1), att2)
        out2 = out2.squeeze(dim=2)
        s_emb.append(out2)

        out = self.decoder(s, r, out, rel, s_emb[-1], t_emb=[])
        return out


class RepeatEncoder(nn.Module):
    def __init__(self, h_dim):
        super(RepeatEncoder, self).__init__()

        self.h_dim = h_dim
        self.num_r = config.num_r
        self.num_e = config.num_e
        self.predict1 = LocalEncoder(h_dim=h_dim, num_layers=config.num_layers)
        self.predict2 = GlobalEncoder(h_dim=h_dim)
        self.rel = EntEmb(use_static_graph=False, self_loop=False, num_e=self.num_r * 2)
        self.ent = EntEmb(use_static_graph=config.use_static_graph, num_e=self.num_e)
        self.tanh = nn.Tanh()
        self.to_one = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.h_dim * 5, self.h_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.h_dim * 3, self.h_dim * 1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.h_dim * 1, 1),
        )
        nn.init.xavier_uniform_(self.to_one[1].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.to_one[4].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.to_one[7].weight, gain=nn.init.calculate_gain('relu'))

        self.union_tim = UnionTimeEncode(h_dim, act=F.tanh)
        self.union_cnt = UnionTimeEncode(h_dim, act=F.tanh)

    def load(self, tag=3):
        model_dict1 = torch.load('./best/{}/local.pt'.format(config.dataset))
        model_dict2 = torch.load('./best/{}/global.pt'.format(config.dataset))
        # self.predict2 = model_dict2['model_state_dict']
        print(tag)
        if tag & 1 == 1:
            print('load model1')
            self.predict1.load_state_dict(model_dict1['model_state_dict'], strict=False)
        if tag & 2 == 2:
            print('load model2')
            self.predict2.load_state_dict(model_dict2['model_state_dict'], strict=False)

    def freeze(self):
        self.predict1.requires_grad_(False)
        self.predict2.requires_grad_(False)

    def get_predict1(self, his_list_g, batch_data, batch_his, batch_cnt):
        return self.predict1(his_list_g, batch_data, batch_his)

    def get_predict2(self, batch_data, cut_t, cnt, neg_t, neg_cnt):
        return self.predict2(batch_data, cut_t, cnt, neg_t, neg_cnt)

    def to_train(self):
        self.train()
        self.predict1.eval()
        self.predict2.eval()

    def forward(self, history, data, score):
        ent = self.ent()
        rel = self.rel()
        ent = F.normalize(ent)
        ent = torch.cat([ent, torch.zeros([1, self.h_dim], device=config.device, requires_grad=False).float()])
        data = torch.tensor(data).to(config.device)
        s, r, o, t = data.T
        score = score.to(config.device)
        neg = history[0].to(config.device)
        cnt = history[1].to(config.device)
        batch, n = neg.size()
        neg = neg.squeeze(dim=1)
        cnt = cnt.squeeze(dim=1)
        neg_emb = ent[neg]

        cc = cnt.clone().float()
        cnt_emb = self.union_cnt(cc, dim=2)

        s_emb = ent[s]
        r_emb = rel[r]
        t_emb = self.union_tim(t)

        s_emb = s_emb.unsqueeze(dim=1).expand_as(cnt_emb)
        r_emb = r_emb.unsqueeze(dim=1).expand_as(cnt_emb)
        t_emb = t_emb.unsqueeze(dim=1).expand_as(cnt_emb)
        out = torch.concat([s_emb, r_emb, t_emb, neg_emb, cnt_emb], dim=2)
        out = self.to_one(out)

        out = out.squeeze(dim=2)

        out = torch.cat([out, score[torch.arange(batch).unsqueeze(1), neg]], dim=1)
        out = F.layer_norm(out, out.size()[1:])
        out = out[:, :n]
        score[torch.arange(batch).unsqueeze(1), neg] += out
        return score
