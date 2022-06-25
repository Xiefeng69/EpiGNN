# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable

from utils import *
from layers import *
from ablation import WOGlobal
from ablation import WOLocal
from ablation import WORAGL
from ablation import baseline

class EpiGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.m = data.m
        self.w = args.window
        self.n_layer = args.n_layer
        self.droprate = args.dropout
        self.hidR = args.hidR
        self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.res = args.res
        self.hw = args.hw
        self.dropout = nn.Dropout(self.droprate)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)

        if args.extra:
            self.extra = True
            self.external = data.external
        else:
            self.extra = False

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = data.degree_adj
        self.s_enc = nn.Linear(1, self.hidR)

        # external resources
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n)])
        #self.GCNBlock1 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
        #self.GCNBlock2 = GraphConvLayer(in_features=self.hidR, out_features=self.hidR)

        # prediction
        if self.res == 0:
            self.output = nn.Linear(self.hidR*2, 1)
        else:
            self.output = nn.Linear(self.hidR*(self.n+1), 1)

        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, index, isEval=False):
        #print(index.shape) batch_size
        batch_size = x.shape[0] # batchsize, w, m

        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(x)

        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb) # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        #attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc

        # step 4: Region-Aware Graph Learner
        # load external resource
        if self.extra:
            extra_adj_list=[]
            zeros_mt = torch.zeros((self.m, self.m)).to(self.adj.device)
            #print(self.external.shape)
            for i in range(batch_size):
                offset = 20
                if i-offset>=0:
                    idx = i-offset
                    extra_adj_list.append(self.external[index[i],:,:].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0) # [1872, 52]
            extra_info = extra_info
            #print(extra_info.shape) # batch_size, self.m self.m
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
            #print(self.external_parameter)

        # apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        
        # if additional information => fusion
        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        # get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)
        '''
        if self.res == 1:
            node_state = torch.cat(node_state_list, dim=-1)
        
        node_state = self.GCNBlock1(feat_emb, laplace_adj)
        node_state = self.dropout(node_state)
        node_state = self.GCNBlock2(node_state, laplace_adj)
        node_state = self.dropout(node_state)
        '''

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway means autoregressive model
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        
        # if evaluation, return some intermediate results
        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        return res, imd
