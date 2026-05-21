# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import json
import numpy as np

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x_feature = self.relu(x)  # 保存进入最后FC层前的特征
        x = self.layer_out(x_feature)
        # [修复] 返回原始 Logits，不经过 softmax
        # nn.CrossEntropyLoss 内部已包含 softmax，外部再加会导致双重 softmax
        return x, x_feature

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x_feature = F.relu(self.fc2(x))  # 保存进入最后FC层前的特征
        x = self.fc3(x_feature)
        # [修复] 返回原始 Logits，去掉 F.log_softmax
        return x, x_feature

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.cls = args.num_classes
        
        # === [新增代码] ===
        self.drop = nn.Dropout(0.5)  # 必须添加这一行，否则 forward 会报错
        # =================

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # === [修复] Dropout 应该在 fc3 之前 ===
        x_feature = self.drop(x)  # 保存进入最后FC层前的特征
        x = self.fc3(x_feature)
        # ===================================
        
        # [修复] 返回原始 Logits，去掉 F.log_softmax
        return x, x_feature

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x_feature = self.drop((F.relu(self.fc2(x))))  # 保存进入最后FC层前的特征
        x = self.fc3(x_feature)
        # [修复] 返回原始 Logits，去掉 F.log_softmax
        return x, x_feature

class CNN_FEMNIST(nn.Module):
    def __init__(self, args):
        super(CNN_FEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, args.num_classes)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x_feature = F.relu(self.fc2(x))  # 保存进入最后FC层前的特征
        x = self.fc3(x_feature)
        # [修复] 返回原始 Logits，去掉 F.log_softmax
        return x, x_feature


class RNNSent(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    def __init__(self,args, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, emb_arr=None):
        super(RNNSent, self).__init__()
        from models.language_utils import get_word_emb_arr  # 仅在需要时导入
        VOCAB_DIR = 'models/embs.json'
        emb, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.encoder = torch.tensor(emb).to(args.device)
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(nhid, 10)
        self.decoder = nn.Linear(10, ntoken)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device = args.device

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = torch.transpose(input, 0,1)
        emb = torch.zeros((25,4,300)) 
        for i in range(25):
            for j in range(4):
                emb[i,j,:] = self.encoder[input[i,j],:]
        emb = emb.to(self.device)
        emb = emb.view(300,4,25)
        self.rnn.flatten_parameters()
        output, new_hidden = self.rnn(emb, hidden)
        
        # 获取特征：取最后一个时间步的输出作为特征表示
        feature = output[-1,:,:] 
        
        output = self.drop(F.relu(self.fc(output)))
        decoded = self.decoder(output[-1,:,:])
        
        # 将新的hidden state保存为实例属性，以便需要时可以访问
        self.last_hidden = new_hidden
        
        # [Fix] 返回 (logits, feature) 元组，保持与其他模型接口一致
        return decoded.t(), feature

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# === 基础残差块 (GroupNorm版本) ===
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # [修改] 使用 GroupNorm (每组32通道，如果通道少于32则每组2通道)
        # 0.5x Slim版最小通道是16，所以 num_groups 设置为 2 或 4 比较安全
        num_groups = 4 if planes % 4 == 0 else 2 
        self.bn1 = nn.GroupNorm(num_groups, planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            num_groups_sc = 4 if (self.expansion*planes) % 4 == 0 else 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups_sc, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# === 瘦身版 ResNet (GroupNorm 版) ===
class ResNet18_CrossFreeze(nn.Module):
    def __init__(self, args, pretrained=False):
        super(ResNet18_CrossFreeze, self).__init__()
        
        # 宽度因子 0.5
        self.width_mult = 0.5 
        
        block = BasicBlock
        num_blocks = [2, 2, 2, 2] 
        num_classes = args.num_classes
        
        # 初始通道数 
        base_channels = int(64 * self.width_mult) # 32
        self.in_planes = base_channels
        
        # --- 初始层 ---
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # [修改] GroupNorm
        self.bn1 = nn.GroupNorm(4, base_channels) 
        self.relu = nn.ReLU(inplace=True)
        
        # --- 残差层 ---
        self.layer1 = self._make_layer(block, int(64 * self.width_mult), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128 * self.width_mult), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 * self.width_mult), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 * self.width_mult), num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Body ---
        self.body = nn.Sequential(
            self.conv1, self.bn1, self.relu,
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.avgpool,
            nn.Flatten()
        )
        
        # --- Heads ---
        final_dim = int(512 * self.width_mult * block.expansion) 
        
        self.fc = nn.Linear(final_dim, num_classes)      
        self.sm_head = nn.Linear(final_dim, num_classes) 

        # 这里的 weight_keys 只是给 FedAvg 一个参考，实际上我们的 main 代码里已经手动处理了
        self.weight_keys = [['sm_head.weight', 'sm_head.bias']]

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mode='local'):
        out = self.body(x)
        if mode == 'local':
            return self.fc(out), out 
        elif mode == 'global':
            return self.sm_head(out), out
        else:
            return self.fc(out)