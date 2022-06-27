# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
from math import sqrt

import scipy.sparse as sp
from scipy.stats.stats import pearsonr
from models import *
from data import *

import stat
import shutil
import logging
import glob
import time
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Training settings
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', type=str, default='region785', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='region-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--n_layer', type=int, default=1, help="number of layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=128, help="batch size")
ap.add_argument('--check_point', type=int, default=1, help="check point")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.5, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.3, help="Testing ratio (0, 1)")
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=False,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=5, help='leadtime default 5')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=0,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--patience', type=int, default=100, help='patience default 100')
ap.add_argument('--k', type=int, default=8,  help='kernels')
ap.add_argument('--hidR', type=int, default=64,  help='hidden dim')
ap.add_argument('--hidA', type=int, default=64,  help='hidden dim of attention layer')
ap.add_argument('--hidP', type=int, default=1,  help='hidden dim of adaptive pooling')
ap.add_argument('--hw', type=int, default=0,  help='highway')
ap.add_argument('--extra', type=str, default='',  help='externel folder')
ap.add_argument('--label', type=str, default='',  help='label_file')
ap.add_argument('--pcc', type=str, default='',  help='have pcc?')
ap.add_argument('--n', type=int, default=2, help='layer number of GCN')
ap.add_argument('--res', type=int, default=0, help='0 means no residual link while 1 means need residual link')
ap.add_argument('--s', type=int, default=2, help='kernel size of temporal convolution network')
ap.add_argument('--result', type=int, default=0, help='0 means do not show result while 1 means show result')
ap.add_argument('--ablation', type=str, default=None, help='ablation test')
ap.add_argument('--eval', type=str, default='', help='evaluation test file')
ap.add_argument('--record', type=str, default='', help='record the result')
ap.add_argument('--model', type=str, default='EpiGNN', help='model')
args = ap.parse_args() 
print('--------Parameters--------')
print(args)
print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available() 
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
logger.info('cuda %s', args.cuda)

time_token = str(time.time()).split('.')[0] # tensorboard model
log_token = '%s.%s.w-%s.h-%s' % (args.model, args.dataset, args.window, args.horizon)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    try:
        shutil.rmtree(tensorboard_log_dir) # remove a folder
    except PermissionError as e:
        err_file_path = str(e).split("\'",2)[1]
        if os.path.exists(err_file_path):
            os.chmod(err_file_path, stat.S_IWUSR) # write by owner: https://blog.csdn.net/cute_boy_/article/details/119926001
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

data_loader = DataBasicLoader(args)

if args.ablation is None:
    model = EpiGNN(args, data_loader)  
else:
    if args.ablation == 'woglobal':
        model = WOGlobal(args, data_loader)
    elif args.ablation =='wolocal':
        model = WOLocal(args, data_loader)
    elif args.ablation == 'woragl':
        model = WORAGL(args, data_loader)
    elif args.ablation == 'baseline':
        model = baseline(args, data_loader)
    else:
        raise LookupError('can not find the model')

logger.info('model %s', model)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:',pytorch_total_params)

def evaluate(data_loader, data, tag='val', show=0):
    model.eval()
    total = 0.
    n_samples = 0.
    total_loss = 0.
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []
    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        output,_  = model(X, index)
        loss_train = F.mse_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        n_samples += (output.size(0) * data_loader.m);

        x_value_mx.append(X.data.cpu())
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx) # [n_samples, 47] 
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  #(#n_samples, 47)
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) # mean of 47
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places 
    if not args.pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            pcc_tmp.append(pearsonr(y_true_states[:,k],y_pred_states[:,k])[0])
        pcc_states = np.mean(np.array(pcc_tmp)) 
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))
    y_pred = np.reshape(y_pred_states,(-1))
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    if show==1:
        print('x vaue', x_value_states)
        print('ground true', y_true.shape)
        print('predict value', y_pred.shape)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 0.00001)))
    mape /= 10000000
    if not args.pcc:
        pcc = pearsonr(y_true,y_pred)[0]
    else:
        pcc=1
        pcc_states=1
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae

def train(data_loader, data):
    model.train()
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        optimizer.zero_grad()
        output,_  = model(X, index) 
        if Y.size(0) == 1:
            Y = Y.view(-1)
        loss_train = F.mse_loss(output, Y) # mse_loss
        total_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return float(total_loss / n_samples)
 
bad_counter = 0
best_epoch = 0
best_val = 1e+20;
try:
    print('begin training');
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val)
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss}, epoch )
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
       
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('Best validation epoch:',epoch, time.ctime());
            test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')
            print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test',show=args.result)
print('Final evaluation')

# record test results.
if args.record != '':
    with open("result/result.txt","a",encoding="utf-8") as f:
        f.write('Model: EpiGNN, dataset: {}, window: {}, horizon: {}, seed: {}, MAE: {:5.4f}, RMSE: {:5.4f}, PCC: {:5.4f}, lr: {}, GNN_number: {}, filter_num: {}, res: {}, hidR: {}, hidP: {}, hidA: {}, dropout: {}\n'.format(args.dataset, args.window, args.horizon, args.seed, mae, rmse, pcc, args.lr, args.n, args.k, args.res, args.hidR, args.hidP, args.hidA, args.dropout))
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format( mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))

# test the trained model.
if args.eval != '':
    testdata = np.loadtxt(open("data/{}.txt".format(args.eval)), delimiter=',')
    testdata = (testdata - data_loader.min) / (data_loader.max - data_loader.min)
    testdata = torch.Tensor(testdata)
    testdata = testdata.unsqueeze(0)
    testdata = Variable(testdata)
    if args.cuda:
        testdata = testdata.cuda()

    model.eval()
    with torch.no_grad():
        out_data, addinfo = model(testdata, None, isEval=True)
        out_data = out_data.cpu().numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min
        # print(out_data.shape) 1, numregion
        out_data = out_data.squeeze(0)

    # record
    out_data = out_data.tolist()
    adjacent = addinfo[0].squeeze(0).cpu().numpy()
    #adjacent = np.around(adjacent, 2)
    adjacent = adjacent.tolist()
    attn = addinfo[1].squeeze(0).cpu().numpy()
    attn = np.around(attn, 2)
    attn = attn.tolist()
    with open("save/{}.txt".format(args.eval+"result"), "a") as f:
        f.write("\n" + "window" + str(args.window) + ", horizon" + str(args.horizon) + "\n")
        f.write(str(out_data))
        f.write('\n')
    with open("save/{}.txt".format(args.eval+"adjacent"), "a") as f:
        f.write("\n" + "window" + str(args.window) + "horizon" + str(args.horizon) + "\n")
        f.write(str(adjacent))
        f.write('\n')
    with open("save/{}.txt".format(args.eval+"attn"), "a") as f:
        f.write("\n" + "window" + str(args.window) + "horizon" + str(args.horizon) + "\n")
        f.write(str(attn))
        f.write('\n')