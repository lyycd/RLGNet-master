import torch
import yaml

import config
import globalEncoder
import repeatEncoder
import localEncoder
import argparse

from utilities import utils
from model import LocalEncoder, RepeatEncoder
from utilities.models import run_valid, run_valid4, get_score, run_valid6

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RLGNet")
    parser.add_argument("--multi_step", action="store_true", default=False)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--train_global', action="store_true", default=False)
    parser.add_argument('--train_local', action="store_true", default=False)
    parser.add_argument('--train_repeat', action="store_true", default=False)
    parser.add_argument('--test_global', action="store_true", default=False)
    parser.add_argument('--test_local', action="store_true", default=False)
    parser.add_argument('--test_repeat', action="store_true", default=False)
    parser.add_argument('--x', default=-1, type=float)
    parser.add_argument('--top_k', default=20, type=int)

    args = parser.parse_args()
    tmpx = args.x
    # print(args)
    # 读取配置文件内容
    with open('utilities/args.yaml', 'r') as file:
        config_file = yaml.safe_load(file)
    # 更新 argparse 的默认值
    for key, value in config_file[args.dataset].items():
        setattr(args, key, value)
    # 假设YAML文件中有一个'settings'代码块，里面包含了超参数配置
    # parameters = config_file[args.dataset]
    config.his_k = 10
    print(args)
    if args.train_global or args.test_global or args.train_repeat or args.test_repeat:
        config.init(data_name=args.dataset, is_mul=args.multi_step, seq_len=args.seq_len, num_top_k=args.top_k)
    else:
        config.init(data_name=args.dataset, is_mul=args.multi_step, seq_len=args.seq_len, global_parameter=False, num_top_k=args.top_k)
    if tmpx != -1:
        args.x = tmpx
    config.x = float(args.x)
    if args.train_global:
        globalEncoder.run()
    if args.train_local:
        localEncoder.run()
    if args.train_repeat:
        repeatEncoder.run(config.x)
    print('load model1')
    print('load model2')
    all_ans_list_test = utils.load_all_answers_for_time_filter(config.test_data, config.num_r, config.num_e, False)

    if args.test_global:
        repeat_model = RepeatEncoder(h_dim=config.h_dim)
        repeat_model.to(config.device)
        repeat_model.load(tag=2)
        repeat_model.freeze()
        test_out = get_score(repeat_model, config.dev_list_g, config.test_list, config.test_list_his, config.test_times,
                             list_cnt=config.test_list_cnt,
                             cut_t=config.test_all_cut_t, data_o_cnt=config.test_all_o_cnt, data_o_t=config.test_all_o_t,
                             cnt=config.test_all_cnt,
                             multi_step=config.multi_step, current_list_g=config.test_list_g, x=0)

        print('----------global----------')
        mrr_raw, mrr_filter = run_valid6(out=test_out, times=config.test_times, all_ans_list=all_ans_list_test,
                                         data_list=config.test_list)
    if args.test_local:
        local_model = LocalEncoder(h_dim=config.h_dim, num_layers=config.num_layers)
        local_model = local_model.to(config.device)
        model_dict = torch.load('./best/{}/local.pt'.format(config.dataset))
        local_model.load_state_dict(model_dict['model_state_dict'], strict=False)
        print('----------local----------')
        mrr_raw, mrr_filter = run_valid(model=local_model, list_g=config.train_list_g + config.dev_list_g, list_data=config.test_list,
                                        times=config.test_times, list_cnt=config.test_list_cnt, multistep=config.multi_step,
                                        current_list_g=config.test_list_g, list_his=config.test_list_his, all_ans_list=all_ans_list_test)

    if args.test_repeat:
        repeat_model = RepeatEncoder(h_dim=config.h_dim)
        repeat_model.to(config.device)
        repeat_model.load()
        repeat_model.freeze()
        model_dict = torch.load('./best/{}/repeat.pt'.format(config.dataset))
        repeat_model.load_state_dict(model_dict['model_state_dict'], strict=False)
        # repeat_model.freeze()

        test_out = get_score(repeat_model, config.dev_list_g, config.test_list, config.test_list_his, config.test_times,
                             list_cnt=config.test_list_cnt,
                             cut_t=config.test_all_cut_t, data_o_cnt=config.test_all_o_cnt, data_o_t=config.test_all_o_t,
                             cnt=config.test_all_cnt,
                             multi_step=config.multi_step, current_list_g=config.test_list_g, x=config.x)
        print('----------repeat----------')
        mrr_raw, mrr_filter = run_valid4(model=repeat_model, data=config.test_data, out=test_out, data_cnt=config.test_cnt,
                                         data_o=config.test_o_cnt, times=config.test_times, all_ans_list=all_ans_list_test,
                                         data_list=config.test_list, is_filter=True)

# args--
