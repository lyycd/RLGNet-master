import copy
import os
import time

from torch import nn
import torch
from tqdm import tqdm

import config
from model import LocalEncoder
from utilities.models import run_valid
import random


# def test():


def run():
    best_mrr = 0
    best_hits1 = 0
    best_hits3 = 0
    best_hits10 = 0
    criterion = nn.CrossEntropyLoss()
    model = LocalEncoder(h_dim=config.h_dim, num_layers=config.num_layers)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    times = copy.copy(config.train_times)
    if config.dataset in ['YAGO', 'WIKI']:
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    else:
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    max_tim = config.train_times[-1]
    for epoch in range(config.max_epoch):
        train_correct = 0
        train_loss = 0
        train_ce_loss = 0
        StepLR.step()

        print('------------------train--------------------')
        time.sleep(0.1)
        model.train()
        random.shuffle(times)
        for tim in tqdm(times):
            if tim == 0:
                continue

            his_list_g = config.train_list_g[max(0, tim - config.his_len):tim]
            batch_data = config.train_list[tim]
            batch_his = config.train_list_his[tim]
            batch_cnt = config.train_list_cnt[tim]
            batch_data = torch.tensor(batch_data)
            labels = batch_data[:, 2].long()

            score = model(his_list_g, batch_data, batch_his)
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
        print('ce_loss={}'.format(train_ce_loss))

        time.sleep(0.1)
        if epoch >= config.valid_epoch:
            print('------------------valid--------------------')
            time.sleep(0.1)
            mrr, hits1, hits3, hits10 = run_valid(model=model, list_g=config.train_list_g, list_data=config.dev_list, times=config.dev_times,
                                                  list_cnt=config.dev_list_cnt, multistep=False, current_list_g=config.dev_list_g,
                                                  list_his=config.dev_list_his)
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
                        os.makedirs('./best/{}'.format(config.dataset))
                    best_mrr = mrr
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, './best/{}/local.pt'.format(config.dataset))
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

# Press the green button in the gutter to run the script.
