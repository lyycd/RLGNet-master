import math

import torch
import torch.nn.functional as F
import config
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, h_dim, drop=0.5, kernel_size=3, channels=50):
        super(Decoder, self).__init__()
        self.h_dim = h_dim
        self.num_e = config.num_e
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.tanh = nn.Tanh()
        self.conv1 = torch.nn.Conv1d(3, out_channels=channels, kernel_size=kernel_size, stride=1,
                                     padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(3)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.2)
        self.feature_map_drop = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.h_dim * channels, self.h_dim)

    def forward(self, s, r, ent, rel, his_emb, r_emb=None, t_emb=None):
        # print(s.size())
        batch_size = s.size(0)
        ent = self.tanh(ent)
        his_emb = self.tanh(his_emb)
        s_emb = ent[s]
        # print(len(t_emb))
        # a, b = t_emb
        if r_emb is None:
            r_emb = rel[r]
        stacked_inputs = torch.stack([s_emb, r_emb, his_emb], 1)

        # print(stacked_inputs.size())
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent.transpose(1, 0))

        return x


class Decoder2(nn.Module):
    def __init__(self, drop=0.5, kernel_size=3, channels=50):
        super(Decoder2, self).__init__()
        self.h_dim = config.h_dim
        self.num_e = config.num_e
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.tanh = nn.Tanh()
        self.conv1 = torch.nn.Conv1d(5, channels, kernel_size=kernel_size, stride=1,
                                     padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(5)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.inp_drop = torch.nn.Dropout(drop)
        self.hidden_drop = torch.nn.Dropout(drop)
        self.feature_map_drop = torch.nn.Dropout(drop)
        self.fc = torch.nn.Linear(self.h_dim * channels, self.h_dim)

    # bu
    def forward(self, s, r, ent, rel, t_emb):
        batch_size = s.size(0)
        ent = self.tanh(ent)
        s_emb = ent[s]
        r_emb = rel[r]
        a, b, c= t_emb
        stacked_inputs = torch.stack([s_emb, r_emb, a, b, c], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent.transpose(1, 0))
        return x

class Decoder3(nn.Module):
    def __init__(self, drop=0.5, kernel_size=3, channels=50):
        super(Decoder3, self).__init__()
        self.h_dim = config.h_dim
        self.num_e = config.num_e
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.tanh = nn.Tanh()
        self.conv1 = torch.nn.Conv1d(5, channels, kernel_size=kernel_size, stride=1,
                                     padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(5)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.inp_drop = torch.nn.Dropout(drop)
        self.hidden_drop = torch.nn.Dropout(drop)
        self.feature_map_drop = torch.nn.Dropout(drop)
        self.fc = torch.nn.Linear(self.h_dim * channels, self.h_dim)

    # bu
    def forward(self, s, r, ent, rel, t_emb):
        batch_size = s.size(0)
        ent = self.tanh(ent)
        s_emb = ent[s]
        r_emb = rel[r]
        a, b, c= t_emb
        stacked_inputs = torch.stack([s_emb, r_emb, a, b, c], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent.transpose(1, 0))
        return x
class DecoderR(torch.nn.Module):
    def __init__(self, drop=0.5):
        super(DecoderR, self).__init__()
        self.h_dim = config.h_dim
        self.num_e = config.num_e
        self.num_r = config.num_r

        self.inp_drop = torch.nn.Dropout(self.h_dim)
        self.hidden_drop = torch.nn.Dropout(self.h_dim)
        self.feature_map_drop = torch.nn.Dropout(0.2)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, 50, 3, stride=1,
                                     padding=int(math.floor(3 / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.fc = torch.nn.Linear(self.h_dim * 2, self.h_dim)
        self.bn3 = torch.nn.BatchNorm1d(self.h_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(self.h_dim)

    def forward(self, s_emb, r_emb, ent):
        batch_size = s_emb.size(0)
        s_emb = self.tanh(s_emb)
        r_emb = self.tanh(r_emb)
        ent = self.tanh(ent)
        stacked_inputs = torch.stack([s_emb, r_emb], 1)
        # print(stacked_inputs.size())
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent.transpose(1, 0))

        return x


class DecoderT(nn.Module):
    def __init__(self, drop=0.5, kernel_size=3, channels=50):
        super(DecoderT, self).__init__()
        self.h_dim = config.h_dim
        self.num_e = config.num_e
        self.mlp = nn.Linear(self.h_dim * 2, self.num_e)
        self.mlp2 = nn.Linear(self.h_dim, self.num_e)
        self.linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlp2.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.conv1 = torch.nn.Conv1d(4, channels, kernel_size=kernel_size, stride=1,
                                     padding=int(math.floor(3 / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.inp_drop = torch.nn.Dropout(drop)
        self.hidden_drop = torch.nn.Dropout(drop)
        self.feature_map_drop = torch.nn.Dropout(drop)
        self.fc = torch.nn.Linear(self.h_dim * channels, self.h_dim)

    # bu
    def forward(self, a, b, c, d):
        # print(s.size())
        # print(s)
        batch_size = a.size(0)
        # ent = F.normalize(ent)
        stacked_inputs = torch.stack([a, b, c, d], 1)
        # print(stacked_inputs.size())
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        return x


class DecoderTT(nn.Module):
    def __init__(self, drop=0.5, kernel_size=3, channels=50):
        super(DecoderTT, self).__init__()
        self.h_dim = config.h_dim
        self.num_e = config.num_e
        self.mlp = nn.Linear(self.h_dim * 2, self.num_e)
        self.mlp2 = nn.Linear(self.h_dim, self.num_e)
        self.linear = nn.Linear(self.h_dim * 2, self.h_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlp2.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.conv1 = torch.nn.Conv1d(config.his_k, channels, kernel_size=kernel_size, stride=1,
                                     padding=int(math.floor(3 / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(config.his_k)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(self.h_dim)
        self.inp_drop = torch.nn.Dropout(drop)
        self.hidden_drop = torch.nn.Dropout(drop)
        self.feature_map_drop = torch.nn.Dropout(drop)
        self.fc = torch.nn.Linear(self.h_dim * channels * 2, self.h_dim)

    # bu
    def forward(self, a):
        # print(s.size())
        # print(s)
        batch_size = a[0].size(0)
        # ent = F.normalize(ent)
        stacked_inputs = torch.stack(a, 1)
        # print(stacked_inputs.size())
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        x = self.hidden_drop(x)

        if batch_size > 1:
            x = self.bn2(x)
        return x
