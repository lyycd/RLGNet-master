import os
import time
import random
import torch
import config
from utilities import utils, models
from model import RepeatEncoder
from torch import optim, nn


def run(x=None):
    best_mrr = 0
    best_hits1 = 0
    best_hits3 = 0
    best_hits10 = 0
    model = RepeatEncoder(h_dim=config.h_dim)
    model.to(config.device)
    model.load()
    model.freeze()

    if x is None:
        config.x = 0.8
    else:
        config.x = x
    lr = config.lr3
    criterion = nn.CrossEntropyLoss()

    train_state = torch.tensor([config.train_data[i, 2] in config.train_o_cnt[i] for i in range(len(config.train_data))])
    train_data = config.train_data[train_state]
    train_data_o = config.train_o_cnt[train_state]
    train_data_cnt = config.train_cnt[train_state]

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-5)

    train_out = models.get_score(model, None, config.train_list, config.train_list_his, config.train_times,
                                 list_cnt=config.train_list_cnt, cut_t=config.train_all_cut_t, data_o_t=config.train_all_o_t,
                                 cnt=config.train_all_cnt,
                                 data_o_cnt=config.train_all_o_cnt, multi_step=False, current_list_g=config.train_list_g, x=config.x,
                                 state=train_state)

    dev_out = models.get_score(model, config.train_list_g, config.dev_list, config.dev_list_his, config.dev_times, list_cnt=config.dev_list_cnt,
                               cut_t=config.dev_all_cut_t, data_o_cnt=config.dev_all_o_cnt, data_o_t=config.dev_all_o_t, cnt=config.dev_all_cnt,
                               multi_step=False,
                               current_list_g=config.dev_list_g,
                               x=config.x)

    test_out = models.get_score(model, config.dev_list_g, config.test_list, config.test_list_his, config.test_times, list_cnt=config.test_list_cnt,
                                cut_t=config.test_all_cut_t, data_o_cnt=config.test_all_o_cnt, data_o_t=config.test_all_o_t, cnt=config.test_all_cnt,
                                multi_step=config.multi_step, current_list_g=config.test_list_g, x=config.x)

    rd_idx = [_ for _ in range(len(train_out))]
    all_ans_list_test = utils.load_all_answers_for_time_filter(config.test_data, config.num_r, config.num_e, False)
    for epoch in range(config.max_epoch3):

        train_correct = 0
        train_loss = 0
        model.to_train()
        random.shuffle(rd_idx)

        for batch_data in utils.make_batch([train_out, train_data_o, train_data_cnt, train_data], batch_size=1024, rd_idx=rd_idx):
            out = batch_data[0]
            data_o = batch_data[1]
            data_cnt = batch_data[2]
            data = batch_data[3]
            data_o = torch.tensor(data_o)
            data_cnt = torch.tensor(data_cnt)
            labels = torch.tensor(data[:, 2]).long().to(config.device)
            score = model([data_o, data_cnt], data, out)
            loss = criterion(score, labels)
            batch_train_correct = (torch.argmax(score.cpu(), dim=1) == labels.cpu()).sum()
            train_correct += batch_train_correct
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: {}'.format(epoch))
        print("train_correct={:.6f}".format(train_correct / (len(train_data))))
        print('train_loss={}'.format(train_loss))

        time.sleep(0.1)
        if epoch >= config.valid_epoch:
            print('------------------valid--------------------')
            time.sleep(0.1)
            mrr, hits1, hits3, hits10 = models.run_valid4(model=model, data=config.dev_data, out=dev_out, data_cnt=config.dev_cnt,
                                                          data_o=config.dev_o_cnt, times=config.dev_times, data_list=config.dev_list)
            print('epoch: {}'.format(epoch))
            print("valid MRR : {:.6f}".format(mrr))
            print("valid Hits @ 1: {:.6f}".format(hits1))
            print("valid Hits @ 3: {:.6f}".format(hits3))
            print("valid Hits @ 10: {:.6f}".format(hits10))
            time.sleep(0.1)
            if epoch >= 0:
                if mrr > best_mrr:
                    tag = 1
                else:
                    tag = 0
                if tag == 1 and config.save_model:
                    best_mrr = mrr
                    model_weight = model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, './best/{}/repeat.pt'.format(config.dataset))

                if hits3 > best_hits3:
                    best_hits3 = hits3
                if hits10 > best_hits10:
                    best_hits10 = hits10
                if hits1 > best_hits1:
                    best_hits1 = hits1

    time.sleep(0.1)
    print('------------------best valid--------------------')
    print("best MRR : {:.6f}".format(best_mrr))
    print("best Hits @ 1: {:.6f}".format(best_hits1))
    print("best Hits @ 3: {:.6f}".format(best_hits3))
    print("best Hits @ 10: {:.6f}".format(best_hits10))

