import os
import time

from torch import nn
import torch
import random
import config
from utilities import utils
from model import GlobalEncoder
from utilities.models import run_valid2


def run():
    best_mrr = 0
    best_hits1 = 0
    best_hits3 = 0
    best_hits10 = 0
    criterion = nn.CrossEntropyLoss()
    model = GlobalEncoder(h_dim=config.h_dim)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    rd_idx = [_ for _ in range(len(config.train_data))]

    for epoch in range(config.max_epoch):
        train_correct = 0
        train_loss = 0
        print('------------------train--------------------')
        time.sleep(0.1)
        model.train()
        random.shuffle(rd_idx)

        for batch_data in utils.make_batch([config.train_data, config.train_all_cut_t, config.train_all_cnt, config.train_all_o_t,
                                            config.train_all_o_cnt], batch_size=1024, rd_idx=rd_idx):
            batch_o_cnt = batch_data[4]
            batch_o_t = batch_data[3]
            batch_cnt = batch_data[2]
            batch_cut_t = batch_data[1]
            batch_data = torch.tensor(batch_data[0])

            score = model(batch_data, batch_cut_t, batch_cnt, batch_o_t, batch_o_cnt)

            labels = batch_data[:, 2].long()
            loss = criterion(score, labels.to(config.device))
            batch_train_correct = (torch.argmax(score.cpu(), dim=1) == labels).sum()
            train_correct += batch_train_correct
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        print('epoch: {}'.format(epoch))
        print("train_correct={:.6f}".format(train_correct / (len(config.train_data))))
        print('train_loss={}'.format(train_loss))
        time.sleep(0.1)
        if epoch >= config.valid_epoch:
            print('------------------valid--------------------')
            time.sleep(0.1)
            mrr, hits1, hits3, hits10 = run_valid2(model=model, data=config.dev_data, cut_t=config.dev_all_cut_t, data_o_cnt=config.dev_all_o_cnt,
                                                   data_o_t=config.dev_all_o_t, cnt=config.dev_all_cnt)
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
                    if not os.path.exists('./best/{}'.format(config.dataset)):
                        # 如果目录不存在，创建它
                        os.makedirs('./best/{}'.format(config.dataset))
                    best_mrr = mrr
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, './best/{}/global.pt'.format(config.dataset))
                if hits1 > best_hits1:
                    best_hits1 = hits1
                if hits3 > best_hits3:
                    best_hits3 = hits3
                if hits10 > best_hits10:
                    best_hits10 = hits10
    time.sleep(0.1)
    print('------------------best valid--------------------')
    print("best MRR : {:.6f}".format(best_mrr))
    print("best Hits @ 1: {:.6f}".format(best_hits1))
    print("best Hits @ 3: {:.6f}".format(best_hits3))
    print("best Hits @ 10: {:.6f}".format(best_hits10))


