import torch
import torch.nn as nn
import tqdm

import config
from utilities import utils
from utilities.evolution import calc_raw_mrr


def run_valid(model, list_g, list_data, times, list_cnt, current_list_g=None, multistep=False, list_his=None, all_ans_list=None):
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    his_list_g = list_g[-config.his_len:]
    model.eval()
    ranks_raw, ranks_filter = [], []
    with torch.no_grad():
        for tim in tqdm.tqdm(times):
            if tim == 0:
                continue
            if config.list_is_empty[tim] == 1:
                # print('empty=', tim)
                continue
            batch_data = list_data[tim - times[0]]
            batch_data = torch.tensor(batch_data)
            batch_his = list_his[tim - times[0]]
            batch_cnt = list_cnt[tim - times[0]]
            score = model(his_list_g, batch_data, batch_his)
            labels = batch_data[:, 2].long()
            loss += criterion(score, labels.to(config.device)).item()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            if all_ans_list is not None:
                mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(batch_data, score.cpu(),
                                                                                        all_ans_list[tim - config.test_times[0]],
                                                                                        eval_bz=1000, rel_predict=0)
                ranks_raw.append(rank_raw)
                ranks_filter.append(rank_filter)

            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n
            if multistep is False:
                his_list_g.pop(0)
                his_list_g.append(current_list_g[tim - times[0]])

    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    if all_ans_list is not None:
        mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
        mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
        return mrr_raw, mrr_filter
    return mrr, hits1, hits3, hits10


def run_valid2(model, data, cut_t, cnt, data_o_t, data_o_cnt):
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch_data in utils.make_batch([data, cut_t, cnt, data_o_t, data_o_cnt], batch_size=4096):
            batch_o_cnt = batch_data[4]
            batch_o_t = batch_data[3]
            batch_cnt = batch_data[2]
            batch_cut_t = batch_data[1]
            batch_data = torch.tensor(batch_data[0])

            empty_state = [config.list_is_empty[t.item()] != 1 for (s, r, o, t) in batch_data]
            batch_o_cnt = batch_o_cnt[empty_state]
            batch_o_t = batch_o_t[empty_state]
            batch_cnt = batch_cnt[empty_state]
            batch_cut_t = batch_cut_t[empty_state]
            batch_data = batch_data[empty_state]
            if len(batch_data) == 0:
                continue
            score = model(batch_data, batch_cut_t, batch_cnt, batch_o_t, batch_o_cnt)
            labels = batch_data[:, 2].long()
            loss += criterion(score, labels.to(config.device)).item()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n
    # print('total=', total)
    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    return mrr, hits1, hits3, hits10


def run_valid3(model, out1, out2, data, dev_times_max):
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch_data in utils.make_batch([out1, out2, data], batch_size=4096):
            batch_out1 = batch_data[0]
            batch_out2 = batch_data[1]
            batch_data = batch_data[2]
            empty_state = [config.list_is_empty[t.item()] != 1 for (s, r, o, t) in batch_data]
            batch_out1 = batch_out1[empty_state]
            batch_out2 = batch_out2[empty_state]
            batch_data = batch_data[empty_state]
            if len(batch_data) == 0:
                continue
            cut_t = batch_data[:, 3] - dev_times_max - 1
            cut_t = cut_t.to(config.device).int()

            score = model(batch_out1, batch_out2, batch_data, cut_t)
            labels = batch_data[:, 2].long()
            loss += criterion(score, labels.to(config.device)).item()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n

    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    return mrr, hits1, hits3, hits10


def get_score(model, list_g, list_data, list_his, times, cut_t, data_o_cnt, data_o_t, cnt, list_cnt, multi_step=True,
              current_list_g=None,
              x=None,
              state=None):
    # print('data_o=', data_o.size())
    # print('data_o[5]', data_o[:5])
    if list_g is not None:
        his_list_g = list_g[-config.his_len:]
    else:
        his_list_g = []
    out, out1, out2, data = [], [], [], []
    pre_i, pre_sum = -1, 0
    if multi_step is False:
        assert (current_list_g is not None), 'current_list_g is None'
    # print('times=', times)
    model.eval()
    with torch.no_grad():
        # print(times)
        for tim in tqdm.tqdm(times):
            batch_data = list_data[tim - times[0]]

            # if config.list_is_empty[tim] == 1:
            #     pre_sum += n
            #     continue
            batch_data_raw = torch.tensor(batch_data)
            batch_his_raw = list_his[tim - times[0]]
            batch_cnt_raw = list_cnt[tim - times[0]]
            out2_tmp = []
            if x == 1:
                out2=[]
            else:
                for batch_packed in utils.make_batch([batch_data_raw, batch_his_raw, batch_cnt_raw], batch_size=2048, use_tqdm=False):
                    batch_data = batch_packed[0]
                    batch_his = batch_packed[1]
                    batch_cnt = batch_packed[2]
                    n = len(batch_data)
                    out2_tmp.append(model.get_predict2(batch_data, cut_t=cut_t[pre_sum:pre_sum + n], cnt=cnt[pre_sum:pre_sum + n],
                                                       neg_cnt=data_o_cnt[pre_sum:pre_sum + n], neg_t=data_o_t[pre_sum:pre_sum + n]).detach().cpu())
                    pre_sum += n
                out2_tmp = torch.cat(out2_tmp)
                out2.append(out2_tmp)
            if tim >= config.his_len:
                if x == 0:
                    out1 = []
                else:
                    out1.append(model.get_predict1(his_list_g, batch_data_raw, batch_his_raw, batch_cnt_raw).detach().cpu())
            else:
                out1.append(torch.zeros_like(out2[-1]).cpu())

            data.append(batch_data)
            if x is not None:
                if x == 1:
                    tmp = out1[-1]
                elif x == 0:
                    tmp = out2[-1]
                else:
                    tmp = x * out1[-1] + (1 - x) * out2[-1]
                tmp = tmp.cpu()
                if state is not None:
                    out_tmp, cc = [], 0
                    for i in range(len(tmp)):
                        pre_i += 1
                        if state[pre_i].item() is True:
                            out_tmp.append(tmp[i])
                        cc += 1
                    del tmp
                    if len(out_tmp) > 0:
                        tmp = torch.stack(out_tmp)
                    else:
                        tmp = []
                if len(tmp) > 0:
                    out.append(tmp)
                # print(out[-1].device)
                out1 = []
                out2 = []
            if multi_step is False:
                if len(his_list_g) >= config.his_len:
                    his_list_g.pop(0)
                his_list_g.append(current_list_g[tim - times[0]])


    if x is not None:
        out = torch.cat(out)
    else:
        out1 = torch.cat(out1)
        out2 = torch.cat(out2)

    if x is not None:
        return out
    return out1, out2


def get_score2(model, list_g, list_data, list_his, times, cut_t, data_o_cnt, data_o_t, cnt, list_cnt, multi_step=True,
               current_list_g=None,
               state=None):
    if list_g is not None:
        his_list_g = list_g[-config.his_len:]
    else:
        his_list_g = []
    out, out1, out2, data = [], [], [], []
    out1_hits1, out2_hits1 = 0, 0
    pre_i, pre_sum = -1, 0
    if multi_step is False:
        assert (current_list_g is not None), 'current_list_g is None'
    # print('times=', times)
    model.eval()
    with torch.no_grad():
        # print(times)
        for tim in tqdm.tqdm(times):
            batch_data = list_data[tim - times[0]]
            # if config.list_is_empty[tim] == 1:
            #     pre_sum += n
            #     continue
            batch_data_raw = torch.tensor(batch_data)
            batch_his_raw = list_his[tim - times[0]]
            batch_cnt_raw = list_cnt[tim - times[0]]
            out2_tmp = []
            for batch_packed in utils.make_batch([batch_data_raw, batch_his_raw, batch_cnt_raw], batch_size=2048, use_tqdm=False):
                batch_data = batch_packed[0]
                n = len(batch_data)
                out2_tmp.append(model.get_predict2(batch_data, cut_t=cut_t[pre_sum:pre_sum + n], cnt=cnt[pre_sum:pre_sum + n],
                                                   neg_cnt=data_o_cnt[pre_sum:pre_sum + n], neg_t=data_o_t[pre_sum:pre_sum + n]).detach().cpu())
                pre_sum += n
            out2_tmp = torch.cat(out2_tmp)
            out2.append(out2_tmp)
            labels = batch_data_raw[:, 2].long()
            if tim >= config.his_len:
                out1.append(model.get_predict1(his_list_g, batch_data_raw, batch_his_raw, batch_cnt_raw).detach().cpu())
            else:
                out1.append(torch.zeros_like(out2[-1]).cpu())

            data.append(batch_data)
            out1_hits1 += (torch.argmax(out1[-1].cpu(), dim=1) == labels).sum()
            out2_hits1 += (torch.argmax(out2[-1].cpu(), dim=1) == labels).sum()
            if multi_step is False:
                if len(his_list_g) >= config.his_len:
                    his_list_g.pop(0)
                his_list_g.append(current_list_g[tim - times[0]])

    out1 = torch.cat(out1)

    out2 = torch.cat(out2)
    if state is not None:
        out1 = out1[state]
        out2 = out2[state]
    return out1, out2


def run_valid4(model, out, data, data_cnt, data_o, times, all_ans_list=None, data_list=None, is_filter=False):
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    ranks_raw, ranks_filter = [], []
    pre_sum = 0

    model.eval()
    with torch.no_grad():
        # for batch_data in utils.make_batch([out, data_o, data_cnt, data], batch_size=4096 * 3):
        for tim in times:
            batch_data = data_list[tim - times[0]]
            batch_data = torch.tensor(batch_data)
            n = len(batch_data)
            if config.list_is_empty[tim] == 1:
                pre_sum += n
                continue

            # print('n=', n)
            batch_out = out[pre_sum:pre_sum + n]
            batch_o = data_o[pre_sum:pre_sum + n]
            batch_cnt = data_cnt[pre_sum:pre_sum + n]
            pre_sum += n

            score = model([batch_o, batch_cnt], batch_data, batch_out)

            labels = batch_data[:, 2].long()
            # loss += criterion(score, labels.to(config.device)).item()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            if all_ans_list is not None and is_filter is True:
                mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(batch_data, score.cpu(),
                                                                                        all_ans_list[tim - config.test_times[0]],
                                                                                        eval_bz=1000, rel_predict=0)
                ranks_raw.append(rank_raw)
                ranks_filter.append(rank_filter)
            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n

    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    if is_filter is True and all_ans_list is not None:
        mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
        mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
        return mrr_raw, mrr_filter
    else:
        return mrr, hits1, hits3, hits10


def run_valid6(out, times, all_ans_list=None, data_list=None):
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    ranks_raw, ranks_filter = [], []
    pre_sum = 0

    with torch.no_grad():
        # for batch_data in utils.make_batch([out, data_o, data_cnt, data], batch_size=4096 * 3):
        for tim in times:
            batch_data = data_list[tim - times[0]]
            batch_data = torch.tensor(batch_data)
            n = len(batch_data)
            if config.list_is_empty[tim] == 1:
                pre_sum += n
                continue

            score = out[pre_sum:pre_sum + n]
            pre_sum += n
            labels = batch_data[:, 2].long()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            if all_ans_list is not None:
                mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(batch_data, score.cpu(),
                                                                                        all_ans_list[tim - config.test_times[0]],
                                                                                        eval_bz=1000, rel_predict=0)
                ranks_raw.append(rank_raw)
                ranks_filter.append(rank_filter)
            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n

    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    if all_ans_list is not None:
        mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
        mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
        return mrr_raw, mrr_filter
    else:
        return mrr, hits1, hits3, hits10


def run_valid7(model, out, data, data_cnt, data_o, times, all_ans_list=None, data_list=None, is_filter=False):
    total = 0
    correct, loss = 0, 0
    mrr, hits1, hits3, hits10 = 0, 0, 0, 0
    ranks_raw, ranks_filter = [], []
    pre_sum = 0
    # out = torch.zeros_like(out)
    model.eval()
    with torch.no_grad():
        # for batch_data in utils.make_batch([out, data_o, data_cnt, data], batch_size=4096 * 3):
        for tim in times:
            batch_data = data_list[tim - times[0]]
            batch_data = torch.tensor(batch_data)
            n = len(batch_data)
            if config.list_is_empty[tim] == 1:
                pre_sum += n
                continue

            # print('n=', n)
            batch_out = out[pre_sum:pre_sum + n]
            batch_o = data_o[pre_sum:pre_sum + n]
            batch_cnt = data_cnt[pre_sum:pre_sum + n]
            pre_sum += n
            score = batch_out
            # score = model([batch_o, batch_cnt], batch_data, batch_out)

            labels = batch_data[:, 2].long()
            # loss += criterion(score, labels.to(config.device)).item()
            batch_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            # print('max=', score.max())
            # 最大值是inf会抛出异常
            # print(score.max())
            assert score.max() != torch.inf, 'the max of score is inf'

            correct += batch_correct
            tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(score.cpu(), labels.cpu(), hits=[1, 3, 10])
            if all_ans_list is not None and is_filter is True:
                mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(batch_data, score.cpu(),
                                                                                        all_ans_list[tim - config.test_times[0]],
                                                                                        eval_bz=1000, rel_predict=0)
                ranks_raw.append(rank_raw)
                ranks_filter.append(rank_filter)
            n = len(batch_data)
            total += n
            mrr += tim_mrr * n
            hits1 += tim_hits1 * n
            hits3 += tim_hits3 * n
            hits10 += tim_hits10 * n

    mrr = mrr / total
    hits1 = hits1 / total
    hits3 = hits3 / total
    hits10 = hits10 / total
    if is_filter is True and all_ans_list is not None:
        mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
        mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
        return mrr_raw, mrr_filter
    else:
        return mrr, hits1, hits3, hits10
