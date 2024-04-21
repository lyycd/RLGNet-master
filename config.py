import time
from collections import defaultdict

import numpy as np
import torch
import tqdm

from utilities import utils

global train_out1, train_out2, dev_out1, dev_out2, test_out1, test_out2
global train_data, train_times, test_data, test_times, dev_data, dev_times
global num_e, num_r, time_stamp, max_train_tim, max_dev_tim, max_test_tim, his_len, max_epoch, max_epoch2, save_model
global h_dim, device, dataset, batch_size, dev_batch_size, lr, lr2, valid_epoch, num_layers, rev
global train_list, dev_list, test_list, train_list_g, dev_list_g, test_list_g
global train_list_his, dev_list_his, test_list_his, dev_cut, train_cut, x, all_top_k
global train_map_list, dev_map_list, test_map_list, max_len, his_k, max_epoch3
global train_o_cnt, train_o_t, train_cnt, dev_o_cnt, dev_o_t, dev_cnt, test_o_cnt, test_o_t, test_cnt, top_k, lr3, multi_step
global train_cut_t, dev_cut_t, test_cut_t, train_all_cut_t, dev_all_cut_t, test_all_cut_t, save_model4
global train_all_cnt, train_all_o_cnt, train_all_o_t, dev_all_cnt, dev_all_o_cnt, dev_all_o_t, test_all_cnt, test_all_o_cnt, test_all_o_t
global train_list_cnt, dev_list_cnt, test_list_cnt, static_emb
global static_graph, static_num_e, static_num_r, use_static_graph, self_graph, top_k_sr, is_init, big_graph_list, list_is_empty


def init_hyperparameter(is_mul, data_name, seq_len, num_top_k):
    global h_dim, device, dataset, batch_size, dev_batch_size, lr, lr2, valid_epoch, num_layers, rev, top_k, lr3, multi_step, top_k_sr, \
        his_len, max_epoch, max_epoch2, save_model, max_len, his_k, max_epoch3, save_model4, x, all_top_k, train_data, train_times, test_data, \
        test_times, dev_data, dev_times, num_e, num_r, time_stamp, max_train_tim, max_dev_tim, max_test_tim, train_list, dev_list, test_list, \
        train_list_g, dev_list_g, test_list_g, static_graph, static_num_e, static_num_r, use_static_graph, self_graph, is_init, list_is_empty

    is_init = True
    h_dim = 200
    device = 'cuda'
    # device = 'cpu'
    if data_name is None:
        # dataset = 'ICEWS18'
        # dataset = 'ICEWS05-15'
        # dataset = 'ICEWS14s'
        # dataset = 'ICEWS14'
        # dataset = 'YAGO'
        dataset = 'GDELT'
        # dataset = 'WIKI'
    else:
        dataset = data_name

    # batch_size = 256
    # batch_size = 512
    # batch_size = 1024
    # batch_size = 2048
    # batch_size = 4096
    batch_size = 8192 * 100
    # dev_batch_size = 1024
    # dev_batch_size = 2048
    dev_batch_size = 4096 * 2 * 2 * 100
    # lr = 0.0005
    # lr = 0.001
    # lr = 0.0015
    lr = 0.001
    # lr = 0.005
    lr2 = 0.001
    lr3 = 0.001
    valid_epoch = 0

    num_layers = 1
    if dataset[:5] == 'ICEWS':
        use_static_graph = True
    else:
        use_static_graph = False

    if seq_len is None:
        his_len = 10
        his_k = 10
    else:
        his_len = seq_len
        his_k = seq_len
    print('use_static_graph=', use_static_graph)
    if is_mul is None:
        # multi_step = False
        multi_step = True
    else:
        multi_step = is_mul
    top_k = num_top_k
    top_k_sr = 500
    all_top_k = 200
    rev = 1
    # rev = 0
    max_epoch = 30
    max_epoch3 = 5

    # num_r *= 2
    # save_model = False
    save_model = True
    save_model4 = True
    train_data, train_times = utils.load_quadruples('./data/{}'.format(dataset), 'train.txt')
    test_data, test_times = utils.load_quadruples('./data/{}'.format(dataset), 'test.txt')
    dev_data, dev_times = utils.load_quadruples('./data/{}'.format(dataset), 'valid.txt')
    num_e, num_r = utils.get_total_number('./data/{}'.format(dataset), 'stat.txt')

    if use_static_graph is True:
        static_num_e, static_num_r, static_graph = utils.load_static('./data/{}'.format(dataset), 'e-w-graph.txt', num_e=num_e)
    self_graph = utils.build_self_graph(num_e)
    time_stamp = train_times[1] - train_times[0]
    max_train_tim = train_times[-1]
    max_dev_tim = dev_times[-1]
    max_test_tim = test_times[-1]
    # print('time_stamp=', time_stamp)
    train_data[:, 3] //= time_stamp
    test_data[:, 3] //= time_stamp
    dev_data[:, 3] //= time_stamp
    train_times //= time_stamp
    test_times //= time_stamp
    dev_times //= time_stamp
    if train_times[0] == 1:
        train_data[:, 3] -= 1
        test_data[:, 3] -= 1
        dev_data[:, 3] -= 1
        train_times -= 1
        test_times -= 1
        dev_times -= 1
    # print(train_times)

    list_is_empty = defaultdict(int)
    train_list = utils.split_by_t(train_data, model='train')
    dev_list = utils.split_by_t(dev_data, list_is_empty=list_is_empty, model='dev')
    test_list = utils.split_by_t(test_data, list_is_empty=list_is_empty, model='test')
    train_times = train_times[np.argsort(train_times)]
    dev_times = dev_times[np.argsort(dev_times)]
    test_times = test_times[np.argsort(test_times)]

    train_data = utils.list2data(train_list)
    dev_data = utils.list2data(dev_list)
    test_data = utils.list2data(test_list)
    train_list_g = [utils.build_graph(g, num_e=num_e, num_r=num_r) for g in train_list]
    dev_list_g = [utils.build_graph(g, num_e=num_e, num_r=num_r) for g in dev_list]
    test_list_g = [utils.build_graph(g, num_e=num_e, num_r=num_r) for g in test_list]
    if rev == 1:
        train_data = utils.add_rev_quadruples(train_data, num_r)
        dev_data = utils.add_rev_quadruples(dev_data, num_r)
        test_data = utils.add_rev_quadruples(test_data, num_r)

        train_list = utils.split_by_t(train_data)
        dev_list = utils.split_by_t(dev_data)
        test_list = utils.split_by_t(test_data)

        train_data = utils.list2data(train_list)
        dev_data = utils.list2data(dev_list)
        test_data = utils.list2data(test_list)


def init(is_mul=None, data_name=None, seq_len=None, global_parameter=True, num_top_k=20, only_test=False):
    global train_list, dev_list, test_list, train_list_g, dev_list_g, test_list_g
    global train_list_his, dev_list_his, test_list_his, dev_cut, train_cut
    global train_map_list, dev_map_list, test_map_list
    global train_o_cnt, train_o_t, train_cnt, dev_o_cnt, dev_o_t, dev_cnt, test_o_cnt, test_o_t, test_cnt
    global train_cut_t, dev_cut_t, test_cut_t, train_all_cut_t, dev_all_cut_t, test_all_cut_t
    global train_all_cnt, train_all_o_cnt, train_all_o_t, dev_all_cnt, dev_all_o_cnt, dev_all_o_t, test_all_cnt, test_all_o_cnt, test_all_o_t
    global train_list_cnt, dev_list_cnt, test_list_cnt, static_emb, big_graph_list

    # 初始化超参数
    init_hyperparameter(is_mul=is_mul, data_name=data_name, seq_len=seq_len, num_top_k=num_top_k)

    train_map_list = utils.get_map(train_list)
    dev_map_list = utils.get_map(dev_list)
    test_map_list = utils.get_map(test_list)

    start_time = time.time()
    all_map_list = train_map_list + dev_map_list + test_map_list
    empty = torch.Tensor([]).detach().int()
    train_list_his, train_list_cnt = [], []
    for data in tqdm.tqdm(train_list):
        x_tmp, y_tmp = utils.get_his(data, all_map_list, empty=empty, top_k_sr=top_k_sr)
        train_list_his.append(x_tmp)
        train_list_cnt.append(y_tmp)
        # print(x.size(), y.size())

    dev_list_his, dev_list_cnt = [], []
    for data in tqdm.tqdm(dev_list):
        x_tmp, y_tmp = utils.get_his(data, all_map_list, empty=empty, top_k_sr=top_k_sr)
        dev_list_his.append(x_tmp)
        dev_list_cnt.append(y_tmp)

    test_list_his, test_list_cnt = [], []
    for data in tqdm.tqdm(test_list):
        if multi_step is False:
            x_tmp, y_tmp = utils.get_his(data, all_map_list, empty=empty, top_k_sr=top_k_sr)
        else:
            x_tmp, y_tmp = utils.get_his(data, train_map_list + dev_map_list, empty=empty, top_k_sr=top_k_sr, last=True)
        test_list_his.append(x_tmp)
        test_list_cnt.append(y_tmp)
    end_time = time.time()
    print('process time is {:.3f}'.format(end_time - start_time))
    print('max_epoch={}'.format(max_epoch))
    start_time = time.time()
    if global_parameter is True:
        (train_all_cnt, train_all_o_cnt, train_all_o_t, train_all_cut_t, dev_all_cnt, dev_all_o_cnt, dev_all_o_t, dev_all_cut_t, test_all_cnt,
         test_all_o_cnt, test_all_o_t, test_all_cut_t), (
            train_cnt, train_o_cnt, train_o_t, train_cut_t, dev_cnt, dev_o_cnt, dev_o_t, dev_cut_t, test_cnt, test_o_cnt, test_o_t, test_cut_t) \
            = utils.run(all_top_k, top_k)
    end_time = time.time()
    # input(train_all_cnt.size())
    print('process time is {:.3f}'.format(end_time - start_time))

