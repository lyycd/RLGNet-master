#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from torch_geometric.data import Data
import copy
from collections import defaultdict
import dgl
import numpy as np
import os
import tqdm
import torch
import config
import torch.nn.functional as F


def get_map(data_list):
    out = []
    for data in data_list:
        tmp = defaultdict(set)
        for (s, r, o, t) in data:
            tmp[(s, r)].add(o)
        out.append(tmp)
    # print(len(out))
    return out


def split_by_t(data, list_is_empty=None, model='train'):
    # data = order_by_t(data)
    out, his = [], defaultdict(list)
    for i in range(data[0][3], data[-1][3] + 1):
        his[i] = []
    for (s, r, o, t) in data:
        his[t].append([s, r, o, t])
    for k, v in his.items():
        # gdelt dataset sometimes is empty
        if len(his[k]) == 0:
            if list_is_empty is not None:
                # print('out=', k)
                list_is_empty[k] = 1
            out.append(copy.copy(out[-1]))
            out[-1][:, 3] = k
            if model == 'train':
                config.train_times = np.concatenate([config.train_times, np.array([k])])
            if model == 'dev':
                config.dev_times = np.concatenate([config.dev_times, np.array([k])])
            if model == 'test':
                config.test_times = np.concatenate([config.test_times, np.array([k])])
            # print('out2=', len(out))
        else:
            out.append(np.array(his[k]))
    return out


def list2data(data_list):
    tmp_list = []
    for data in data_list:
        if len(data) > 0:
            tmp_list.append(data)
    out = np.concatenate(tmp_list)
    return out


def add_rev_quadruples(data, num_r):
    if data.shape[-1] == 4:
        s, r, o, t = data.T
    else:
        s, r, o = data.T
    u = np.concatenate([s, o])
    r = np.concatenate([r, r + num_r])
    v = np.concatenate([o, s])
    if data.shape[-1] == 4:
        t = np.concatenate([t, t])
        out = np.stack([u, r, v, t], axis=1)
    else:
        out = np.stack([u, r, v], axis=1)

    return out


def load_static(inPath, fileName, num_e):
    u, v, r = [], [], []
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            u.append(head)
            v.append(tail)
            r.append(rel)

    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    static_num_e = len(np.unique(v))
    static_num_r = len(np.unique(r))
    v += num_e
    u, v = np.concatenate((u, v)), np.concatenate((v, u))
    r = np.concatenate((r, r + static_num_r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r)

    if len(u) == 0:
        g = dgl.DGLGraph()
    else:
        g = dgl.graph((u, v), num_nodes=num_e + static_num_e)
        g.edata[dgl.ETYPE] = r
        norm = comp_deg_norm(g)
        g.ndata.update({'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    return static_num_e, static_num_r, g


def build_self_graph(num_e):
    u, v, r = torch.arange(num_e), torch.arange(num_e), torch.zeros(num_e)

    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    v += num_e
    u, v = np.concatenate((u, v)), np.concatenate((v, u))
    r = np.concatenate((r, r + 1))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r)

    if len(u) == 0:
        g = dgl.DGLGraph()
    else:
        g = dgl.graph((u, v), num_nodes=num_e * 2)
        g.edata[dgl.ETYPE] = r
        norm = comp_deg_norm(g)
        g.ndata.update({'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    return g


def _get_his_(Q, data_map_list, top_k_sr, empty, last=False):
    s, r, o, t = Q.T
    band = np.arange(max(0, t - config.his_k), t) if not last else np.arange(-config.his_k, 0)
    out_tmp = [torch.tensor(list(data_map_list[i][(s, r)])[:top_k_sr], dtype=int, requires_grad=False) if (s, r) in data_map_list[i] else empty for i
               in band]
    out_cnt = [] * len(out_tmp)
    return out_tmp, out_cnt


def get_his(data, data_map_list, top_k_sr, empty, last=False):
    return zip(*(_get_his_(Q, data_map_list, top_k_sr, empty=empty, last=last) for Q in data))


def build_graph(data, num_e, num_r):
    if len(data) == 0:
        g = dgl.DGLGraph()
        num_nodes = num_e
        g.add_nodes(num_nodes)
        return g
    if data.shape[-1] == 4:
        s, r, o, t = data.T
    else:
        s, r, o = data.T
    u = np.concatenate([s, o])
    w = np.concatenate([r, r + num_r])
    v = np.concatenate([o, s])

    g = dgl.DGLGraph()
    num_nodes = num_e
    g.add_nodes(num_nodes)
    g.add_edges(u, v)
    g.edata[dgl.ETYPE] = torch.tensor(w)
    return g


def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    times = list(times)
    times.sort()
    quadrupleList = np.asarray(quadrupleList)
    return quadrupleList, np.asarray(times)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    # print('min=', in_deg.min())
    norm = 1.0 / in_deg
    return norm


def load_static(inPath, fileName, num_e):
    u, v, r = [], [], []
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            u.append(head)
            v.append(tail)
            r.append(rel)

    u = np.array(u)
    v = np.array(v)
    r = np.array(r)
    static_num_e = len(np.unique(v))
    static_num_r = len(np.unique(r))
    v += num_e
    u, v = np.concatenate((u, v)), np.concatenate((v, u))
    r = np.concatenate((r, r + static_num_r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r)
    if len(u) == 0:
        g = dgl.DGLGraph()
    else:
        g = dgl.graph((u, v), num_nodes=num_e + static_num_e)
        g.edata[dgl.ETYPE] = r
        norm = comp_deg_norm(g)
        g.ndata.update({'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})

    return static_num_e, static_num_r, g


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def make_batch(data_list, batch_size, rd_idx=None, use_tqdm=True):
    n, m = len(data_list[0]), len(data_list)
    if use_tqdm is True:
        for i in tqdm.tqdm(range(0, n, batch_size)):
            out = []
            for j in range(m):
                if rd_idx is None:
                    result = data_list[j][i:i + batch_size]
                else:
                    # print('j=', j)
                    # print(data_list[j].size())
                    result = data_list[j][rd_idx[i:i + batch_size]]
                out.append(result)
            # print('out=', out[1].size())
            yield out
    else:
        for i in range(0, n, batch_size):
            out = []
            for j in range(m):
                if rd_idx is None:
                    result = data_list[j][i:i + batch_size]
                else:
                    # print('j=', j)
                    # print(data_list[j].size())
                    result = data_list[j][rd_idx[i:i + batch_size]]
                out.append(result)
            # print('out=', out[1].size())
            yield out


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, tim = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict=0):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]
        target = test_triples[batch_start:batch_end, 2]
        rank.append(sort_and_rank(score_batch, target))

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1  # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t:  # 同一时刻发生的三元组
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train)

    # 加入最后一个shapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:, 0], snapshot[:, 2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
    print(
        "# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
        .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]),
                min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r + num_rel in d[e2]:
        d[e2][r + num_rel] = set()
    d[e2][r + num_rel].add(e1)


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)
    out = []
    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    out.append(mrr.item())
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        out.append(avg_count.item())
    return out


def list2tensor(his_o_cnt, his_o_t, his_cnt, cut_t):
    his_o_cnt = torch.nn.utils.rnn.pad_sequence([torch.tensor(_, dtype=torch.int32, requires_grad=False) for _ in his_o_cnt], batch_first=True,
                                                padding_value=-1).detach()
    his_o_t = torch.nn.utils.rnn.pad_sequence([torch.tensor(_, dtype=torch.int32, requires_grad=False) for _ in his_o_t], batch_first=True,
                                              padding_value=-1).detach()
    his_cnt = torch.nn.utils.rnn.pad_sequence([torch.tensor(_, dtype=torch.float32, requires_grad=False) for _ in his_cnt], batch_first=True,
                                              padding_value=-10).detach()
    cut_t = torch.nn.utils.rnn.pad_sequence([torch.tensor(_, dtype=torch.float32, requires_grad=False) for _ in cut_t], batch_first=True,
                                            padding_value=10000).detach()
    return his_o_cnt, his_o_t, his_cnt, cut_t


def get_sr2(data, top_k, pre_cnt=None, pre_cut_times=None, insert=True):
    cut_t, his_o_cnt, his_o_t, his_cnt, cache, pre = [], [], [], [], [], 0
    empty = []
    if pre_cnt is None:
        cnt = defaultdict(lambda: defaultdict(int))
    else:
        cnt = pre_cnt

    if pre_cut_times is None:
        cut_times = defaultdict(list)
    else:
        cut_times = pre_cut_times

    for (s, r, o, t) in tqdm.tqdm(data):
        if pre != t and insert is True:
            for (s_c, r_c, o_c, t_c) in cache:
                cnt[(s_c, r_c)][o_c] += 1
                if len(cut_times[(s_c, r_c)]) >= top_k:
                    cut_times[(s_c, r_c)].pop(0)
                    cut_times[(s_c, r_c)].append([o_c, t_c])
                else:
                    cut_times[(s_c, r_c)].append([o_c, t_c])

            cache = []
        pre = t
        if config.list_is_empty[t] == 0:
            cache.append([s, r, o, t])
        tmp = list(cnt[(s, r)].items())
        if not tmp:
            his_o_cnt.append(empty)
            his_cnt.append(empty)
            cut_t.append(empty)
            his_o_t.append(empty)
        else:
            tmp = np.array(tmp)
            tmp_his_o_t, tmp_cut_t = np.array(cut_times[(s, r)]).T
            his_o_t.append(tmp_his_o_t)
            cut_t.append(tmp_cut_t - t)
            tmp = tmp[np.argsort(-tmp[:, 1])]
            his_o_cnt.append(tmp[:top_k, 0])
            his_cnt.append(tmp[:top_k, 1])

    if insert is True:
        for (s_c, r_c, o_c, t_c) in cache:
            cnt[(s_c, r_c)][o_c] += 1
            if len(cut_times[(s_c, r_c)]) >= top_k:
                cut_times[(s_c, r_c)].pop(0)
                cut_times[(s_c, r_c)].append([o_c, t_c])
            else:
                cut_times[(s_c, r_c)].append([o_c, t_c])

    his_o_cnt, his_o_t, his_cnt, cut_t = list2tensor(his_o_cnt, his_o_t, his_cnt, cut_t)
    return cnt, cut_times, his_o_cnt, his_o_t, his_cnt, cut_t


def select2(his_o_cnt, his_o_t, his_cnt, cut_t, top_k):
    return [his_o_cnt[:, :top_k], his_o_t[:, :top_k], his_cnt[:, :top_k], cut_t[:, top_k]]


# top_k1>=top_k2
def run(top_k1, top_k2):
    if top_k1 < top_k2:
        print('top_k is error')
    train_data = config.train_data
    test_data = config.test_data
    dev_data = config.dev_data

    train_cnt, train_cut_times, train_his_o_cnt1, train_his_o_t1, train_his_cnt1, train_cut_t1 = get_sr2(train_data, top_k=top_k1)
    dev_cnt, dev_cut_times, dev_his_o_cnt1, dev_his_o_t1, dev_his_cnt1, dev_cut_t1 = get_sr2(dev_data, pre_cnt=train_cnt,
                                                                                             pre_cut_times=train_cut_times,
                                                                                             top_k=top_k1)
    train_his_o_cnt2, train_his_o_t2, train_his_cnt2, train_cut_t2 = select2(train_his_o_cnt1, train_his_o_t1, train_his_cnt1, train_cut_t1, top_k2)
    dev_his_o_cnt2, dev_his_o_t2, dev_his_cnt2, dev_cut_t2 = select2(dev_his_o_cnt1, dev_his_o_t1, dev_his_cnt1, dev_cut_t1, top_k2)

    if config.multi_step is False:
        test_cnt, test_cut_times, test_his_o_cnt1, test_his_o_t1, test_his_cnt1, test_cut_t1 = get_sr2(test_data, pre_cnt=dev_cnt,
                                                                                                       pre_cut_times=dev_cut_times,
                                                                                                       top_k=top_k1)
    else:
        test_cnt, test_cut_times, test_his_o_cnt1, test_his_o_t1, test_his_cnt1, test_cut_t1 = get_sr2(test_data, pre_cnt=dev_cnt,
                                                                                                       pre_cut_times=dev_cut_times,
                                                                                                       insert=False, top_k=top_k1)
    test_his_o_cnt2, test_his_o_t2, test_his_cnt2, test_cut_t2 = select2(test_his_o_cnt1, test_his_o_t1, test_his_cnt1, test_cut_t1, top_k2)

    return ((train_his_cnt1, train_his_o_cnt1, train_his_o_t1, train_cut_t1, dev_his_cnt1, dev_his_o_cnt1, dev_his_o_t1, dev_cut_t1, test_his_cnt1,
             test_his_o_cnt1, test_his_o_t1, test_cut_t1),
            (train_his_cnt2, train_his_o_cnt2, train_his_o_t2, train_cut_t2, dev_his_cnt2, dev_his_o_cnt2, dev_his_o_t2, dev_cut_t2, test_his_cnt2,
             test_his_o_cnt2, test_his_o_t2, test_cut_t2))
