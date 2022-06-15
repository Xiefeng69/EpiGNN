# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from utils import *
from torch.autograd import Variable
import sys
import math

class ConvBranch(nn.Module):
    def __init__(self,
                 m,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_factor,
                 hidP=1,
                 isPool=True):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), dilation=(dilation_factor,1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        #self.activate = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x

class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=1, hidP=self.hidP)
        self.conv_l2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=1, hidP=self.hidP)
        self.conv_p1 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=3, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_p2 = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=5, dilation_factor=dilation_factor, hidP=self.hidP)
        self.conv_g = ConvBranch(m=self.m, in_channels=1, out_channels=self.k, kernel_size=self.P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        batch_size = x.shape[0]
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))

        return adj

def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj>0, o_mat, adj)
    d_mat = torch.sum(adj, dim=2) # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    d_mat = torch.pow(d_mat, -0.5)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class WOGlobal(nn.Module):
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

        # local spatial encoding
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

        # feature embedding
        #temp_emb = self.rac(x)
        temp_emb = self.backbone(x)

        # local transmission risk
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # fusion
        feat_emb = temp_emb + s_enc

        # additional information
        if self.extra:
            extra_adj_list=[]
            zeros_mt = torch.zeros((self.m, self.m))
            #print(self.external.shape)
            for i in range(batch_size):
                offset = 20
                if i-offset>=0:
                    idx = i-offset
                    extra_adj_list.append(self.external[index[i],:,:].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0) # [1872, 52]
            #print(extra_info.shape) # batch_size, self.m self.m
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
            #print(self.external_parameter)

        # Graph Learner
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        # additional information
        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
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

        # prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        #res=torch.sigmoid(res)
        # 输出中间信息
        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        return res, imd

class WOLocal(nn.Module):
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
        self.degree = data.degree_adj
        self.dropout = nn.Dropout(self.droprate)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)

        if args.extra:
            self.extra = True
            self.external = data.external
        else:
            self.extra = False

        # Feature embedding
        #self.lstm = nn.LSTM(1, self.hidR, bidirectional=False, batch_first=True, num_layers=self.num_layers, dropout=self.droprate)
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        #self.backbone = TemporalConvNet(num_inputs=self.w, num_channels=[self.hidR]*3, kernel_size=self.s, dropout=self.droprate)

        # global transmission risk
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # external information
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
        #nn.init.orthogonal(self.lstm.weight_ih_l0)
        #nn.init.orthogonal(self.lstm.weight_hh_l0)
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, index, isEval=False):
        #print(index.shape) batch_size
        batch_size = x.shape[0] # batchsize, w, m

        # feature embedding
        '''
        r = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.lstm(r, None)
        last_hid = r_out[:,-1,:]
        temp_emb = last_hid.view(-1,self.m, self.hidR)
        '''
        #temp_emb = self.rac(x)
        temp_emb = self.backbone(x)
        #temp_emb = temp_emb.permute(0, 2, 1)

        # global transmission risk
        query = self.WQ(temp_emb)
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        #attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # fusion
        d = self.degree.unsqueeze(1)
        feat_emb = temp_emb + t_enc

        # additional information
        if self.extra:
            extra_adj_list=[]
            zeros_mt = torch.zeros((self.m, self.m))
            #print(self.external.shape)
            for i in range(batch_size):
                offset = 20
                if i-offset>=0:
                    idx = i-offset
                    extra_adj_list.append(self.external[index[i],:,:].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0) # [1872, 52]
            #print(extra_info.shape) # batch_size, self.m self.m
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)
            #print(self.external_parameter)

        # Graph Learner
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)
        # additional information
        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
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

        # prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        #res=torch.sigmoid(res)
        # 输出中间信息
        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        return res, imd

class DotAtt(nn.Module):
    def __init__(self, attn_dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = self.dropout(attn)
        attn = self.softmax(attn)
        return attn

class WORAGL(nn.Module):
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

        # global transmission risk
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local transmission risk
        self.degree = data.degree_adj
        self.s_enc = nn.Linear(1, self.hidR)

        # external information
        self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # Graph Generator and GCN
        self.query = nn.Linear(self.hidR, self.hidR)
        self.key = nn.Linear(self.hidR, self.hidR)
        self.dotattn = DotAtt()
        self.GNNBlocks = nn.ModuleList([GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n)])

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

        # feature embedding
        #temp_emb = self.rac(x)
        temp_emb = self.backbone(x)
        #temp_emb = temp_emb.permute(0, 2, 1)

        # global transmission risk
        query = self.WQ(temp_emb)
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = self.leakyrelu(attn)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        #attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # local transmission risk
        # print(self.degree.shape) [self.m]
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # fusion
        feat_emb = temp_emb + t_enc + s_enc

        # self-attention
        query = self.query(feat_emb)
        key = self.key(feat_emb)
        adj = self.dotattn(query, key)
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)
        
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)

        # prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z
        #res=torch.sigmoid(res)
        # 输出中间信息
        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        return res, imd

class baseline(nn.Module):
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

        # Feature embedding
        self.hidR = self.k*4*self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        #self.backbone = TemporalConvNet(num_inputs=self.w, num_channels=[self.hidR]*3, kernel_size=self.s, dropout=self.droprate)

        # prediction
        self.output = nn.Linear(self.hidR, 1)


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
        temp_emb = self.backbone(x)
        res = self.output(temp_emb).squeeze(2)
        imd=None
        return res, imd