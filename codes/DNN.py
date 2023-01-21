'''
Created on 8 Apr 2021

@author: aliv
'''
import metrics

import pickle
import time
import os
import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from data_util import read_pkl


def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def log_epoch(epoch, epochs):
    if epoch < 0:
        return False
    if epoch == epochs - 1 or epoch < 5:
        return True
    steps = [[100,5],[500,10],[1000,20],[10000,50]]
    for step, div in steps:
        if epoch < step and (epoch+1) % div == 0:
            return True
    return False



def evaluate(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.tefm, dataset.tedlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.tedlr
        results['lv'] = dataset.telv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results, 
                        f)
    metric = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),y_pred)

    return metric.NDCG(10)

def evaluate_valid(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.vafm, dataset.vadlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.vadlr
        results['lv'] = dataset.valv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results,
                        f)

    metric = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),y_pred)

    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10)

def evaluate_train(dataset, predictor, saveto=None):
    y_pred = predictor(dataset.trfm, dataset.trdlr)
    if saveto is not None:
        results = {}
        results['dlr'] = dataset.trdlr
        results['lv'] = dataset.trlv
        results['pred'] = y_pred
        with open(saveto, 'wb') as f:
            pickle.dump(results,
                        f)

    metric = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),y_pred)
    return metric.NDCG(1), metric.NDCG(3), metric.NDCG(5), metric.NDCG(10)



class LTRData(Dataset):
    def __init__(self, allfm, alllv):
        dev = get_torch_device()
        
        if torch.cuda.is_available():
          self.torch_ = torch.cuda
        else:
          self.torch_ = torch
          
        self.features = self.torch_.FloatTensor(allfm, device=dev)
        self.labels = self.torch_.FloatTensor(alllv, device=dev)
    
    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=None):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i], bias=False))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def save(self, path_to_model):
        torch.save(self.state_dict(), path_to_model)
      
    def train_batch(self, loss_fn, x, y):
        self.opt.zero_grad()
        out = self(x)#[:,0]
        loss = loss_fn(out, y)
        loss.backward()
        self.opt.step()
        return out


def reshape_predicted(predicted, dlr, sessions):
    max_list_size = np.max(np.diff(dlr))
    reshaped = []
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        extended = predicted[s_i:e_i][sessions[qid]]
#         padded = np.pad(extended, ((0,0,),(0,max_list_size-extended.shape[1],),), 'constant', constant_values=0.)
        padded = np.pad(extended, ((0,0,),(0,max_list_size-extended.shape[1],),), 'constant', constant_values=np.nan)
        reshaped.append(padded)
    return np.concatenate(reshaped, 0)

def stack_padded_clicks(clicks, max_list_size):
    stacked = []
    for qid in range(len(clicks)):
#         padded = np.pad(clicks[qid], ((0,0,),(0,max_list_size-clicks[qid].shape[1],),), 'constant', constant_values=0.)
        padded = np.pad(clicks[qid], ((0,0,),(0,max_list_size-clicks[qid].shape[1],),), 'constant', constant_values=np.nan)
        stacked.append(padded)
        
    return np.concatenate(stacked, 0)

def extend_group_ids(group_ids, sessions):
    repeats = [x.shape[0] for x in sessions]
    return np.repeat(group_ids, repeats)


class QueryGroupedLTRData(Dataset):
    def __init__(self, fm, dlr):
        self.feature = fm
        self.dlr = dlr
        self.predicted = np.empty((fm.shape[0],))
        self.lv = np.empty((fm.shape[0],))
        self.dev = get_torch_device()
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
            
    def update_labels(self, labels, qids):
        for qid, label in zip(qids, labels):
            s_i, e_i = self.dlr[qid:qid+2]
            self.lv[s_i:e_i] = label
        
    def update_predicted(self, ys, qids):
        for qid, y in zip(qids, ys):
            s_i, e_i = self.dlr[qid:qid+2]
            self.predicted[s_i:e_i] = y.cpu().data.squeeze()
            
    def __len__(self):
        return self.dlr.shape[0] - 1
    
    def __getitem__(self, qid):
        s_i, e_i = self.dlr[qid:qid+2]
        feature = self.torch_.FloatTensor(self.feature[s_i:e_i,:], device=self.dev)
        lv = self.torch_.FloatTensor(self.lv[s_i:e_i], device=self.dev)
        qids = self.torch_.IntTensor([qid], device=self.dev)
        return feature, lv, qids

def sigmoid(x):
    xx = np.clip(x, -10, 10)
    s = np.where(x >= 0, 
                    1 / (1 + np.exp(-xx)), 
                    np.exp(xx) / (1 + np.exp(xx)))
    s[x > 10] = 1.
    s[x < -10] = 0.
    return s

def normalize(x):
#     y = x - np.nanmin(x, 1)[:,None]
#     s = np.nanmax(x, 1) - np.nanmin(x, 1) + 1.e-6
#     x = y / s[:,None]
#     return sigmoid((x * 10) - 5)
    return sigmoid(x)
    
class Correction():
    def __init__(self, correction, EM_step_size = 0.1):
        self._EM_step_size = EM_step_size
        self.propensity = None
        if correction == 'affine':
            self.expmax = self._affine_expectation
            self.debias = self._affine_debias
        elif correction == 'IPS':
            self.expmax = self._IPS_expectation
            self.debias = self._IPS_debias
        elif correction == 'oracle':
            self.group_ids = None
            self.expmax = self._oracle_expectation
            self.debias = self._affine_debias
        elif correction != 'naive':
            raise Exception('correction method not implemented!')
            
    def load_oracle_values(self, params):
        if self.propensity is None:
            self.propensity = np.ones(self.params_shape) * 0.9
            self.epsilon_p = np.ones(self.params_shape) * 0.9
            self.epsilon_n = np.ones(self.params_shape) * 0.1
            
        for i in range(self.propensity.shape[0]):
            self.propensity[i,:] = np.array(params[str(self.group_ids[i])]['propensity'])[:self.propensity.shape[1]]
            self.epsilon_p[i,:] = np.array(params[str(self.group_ids[i])]['epsilon_p'])[:self.epsilon_p.shape[1]]
            self.epsilon_n[i,:] = np.array(params[str(self.group_ids[i])]['epsilon_n'])[:self.epsilon_n.shape[1]]
        
        
    def _oracle_expectation(self, predicted, clicks, sessions, dlr, group_ids):
        if self.group_ids is None:
            predicted = reshape_predicted(predicted, dlr, sessions)
            self.params_shape = predicted.shape

            self.group_ids = group_ids
        
    def _affine_expectation(self, predicted, clicks, sessions, dlr, group_ids):
        predicted = reshape_predicted(predicted, dlr, sessions)
        labels = stack_padded_clicks(clicks, predicted.shape[1])
        
        self.group_ids = group_ids
#         self.group_ids = extend_group_ids(group_ids, sessions)
        
        if self.propensity is None:
            self.propensity = np.ones_like(labels) * 0.9
            self.epsilon_p = np.ones_like(labels) * 0.9
            self.epsilon_n = np.ones_like(labels) * 0.1
        gamma = normalize(predicted)
        c_prob = (self.epsilon_p * gamma) + (self.epsilon_n * (1. - gamma))

        p_e1_r1_c1 = (self.epsilon_p * gamma) / c_prob
        p_e1_r1_c0 = self.propensity * \
            (1. - self.epsilon_p) * (gamma) / \
            (1 - self.propensity * c_prob)

        p_e1_r0_c1 = 1. - p_e1_r1_c1
        p_e1_r0_c0 = self.propensity * \
            (1. - self.epsilon_n) * (1 - gamma) / \
            (1 - self.propensity * c_prob)

        p_e0_r1_c1 = p_e0_r0_c1 = 0
        p_e0_r1_c0 = (1 - self.propensity) * gamma / \
            (1 - self.propensity * c_prob)
        p_e0_r0_c0 = (1 - self.propensity) * (1 - gamma) / \
            (1 - self.propensity * c_prob)

        
        propensity = (1 - self._EM_step_size) * self.propensity
        epsilon_p = (1 - self._EM_step_size) * self.epsilon_p
        epsilon_n = (1 - self._EM_step_size) * self.epsilon_n
        
        
        propensity_mat = labels + (1 - labels) * (p_e1_r0_c0 + p_e1_r1_c0)

        epsilon_p_mat_nom = labels * p_e1_r1_c1
        epsilon_p_mat_denom = (labels * p_e1_r1_c1) + ((1. - labels) * p_e1_r1_c0)
        epsilon_p_mat_denom[epsilon_p_mat_denom < 1.e-6] = 1.

        epsilon_n_mat_nom = labels * p_e1_r0_c1
        epsilon_n_mat_denom = (labels * p_e1_r0_c1) + ((1. - labels) * p_e1_r0_c0)
        epsilon_n_mat_denom[epsilon_n_mat_denom < 1.e-6] = 1.
        
        unique_groups = np.unique(group_ids)
        
        for group in unique_groups:
            mask = self.group_ids == group
            propensity[mask,:] += self._EM_step_size * np.nanmean(propensity_mat[mask,:], axis=0, keepdims=True)
            epsilon_p[mask,:] += self._EM_step_size * np.nanmean(epsilon_p_mat_nom[mask,:], axis=0, keepdims=True) /\
                                                      np.nanmean(epsilon_p_mat_denom[mask,:], axis=0, keepdims=True)
            epsilon_n[mask,:] += self._EM_step_size * np.nanmean(epsilon_n_mat_nom[mask,:], axis=0, keepdims=True) /\
                                                      np.nanmean(epsilon_n_mat_denom[mask,:], axis=0, keepdims=True)
        
        self.propensity = propensity
        self.epsilon_p = epsilon_p
        self.epsilon_n = epsilon_n
        
        
        def p_r1(clicks, sessions, big_list_index):
            y = clicks + 0.
            inv_index = np.argsort(sessions, axis=1)
            for i in range(sessions.shape[0]):
                y[i,:] = y[i,inv_index[i]]
            session_p_r1 = y * (p_e1_r1_c1[big_list_index, :y.shape[1]]) + (1 - y) * (p_e1_r1_c0 + p_e0_r1_c0)[big_list_index, :y.shape[1]]
            
            return np.clip(session_p_r1.mean(0), 0, 1)
        
        return p_r1
    
    def _affine_debias(self, clicks, sessions, biglist_index):
        beta = self.propensity[biglist_index,:] * self.epsilon_n[biglist_index,:]
        alpha = self.propensity[biglist_index,:] * self.epsilon_p[biglist_index,:] - beta
        
        beta = beta[:, :sessions.shape[1]]
        alpha = alpha[:, :sessions.shape[1]]
        alpha[alpha <= 1.e-6] = 1000. # relevant and non-relevant not distinguishable -> assume non relevant

        gamma = (clicks - beta) / alpha
            
        inv_index = np.argsort(sessions, axis=1)
        for i in range(sessions.shape[0]):
            gamma[i,:] = gamma[i,inv_index[i]]
        gamma = gamma.mean(axis=0)
        return np.clip(gamma, 0, 1)
        
    def _IPS_debias(self, clicks, sessions, qid):
        pass
            
        
    def _IPS_expectation(self, predicted, labels, dlr, group_ids):
        pass
    
    def get_params(self):
        params = {}
        if self.propensity is not None:
            unique_groups = np.unique(self.group_ids)
            for group in unique_groups:
                group_id = np.where(self.group_ids==group)[0][0]
                params[str(group)] = {
                    'propensity': list(self.propensity[group_id, :]),
                    'epsilon_p': list(self.epsilon_p[group_id, :]),
                    'epsilon_n': list(self.epsilon_n[group_id, :]),
                    'count': len(np.where(self.group_ids==group)[0])
                }
        return params


def set_seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed) 

def rbem(jobid, 
        dataset_name, dataset, correction_method,
        epochs, layers,  learning_rate, dropout, rseed, 
        results_file, 
        bernoulli, label_transformation,
        verbose=False):
    results_file = os.path.join(results_file, jobid + '_rbem.json')

    set_seed(rseed)
    
    
    qgdata = QueryGroupedLTRData(dataset.trfm, dataset.trdlr)
    train_dl = DataLoader(qgdata, batch_size=1, shuffle=False)
#     train_dl = DataLoader(qgdata, batch_size=1, shuffle=True)

    net = DNN(dataset.trfm.shape[1], layers, dropout)
    if torch.cuda.is_available():
        net.cuda(get_torch_device())
    print(net)
    
    train_clicks = 0
    for c in dataset.clicks:
        train_clicks += c.sum()
    
    net.opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    correction_op = Correction(correction_method, 0.2)
    
    pr1 = correction_op.expmax(np.ones(dataset.trfm.shape[0])*0.5, 
                               dataset.clicks, dataset.sessions, dataset.trdlr, dataset.group_ids)
    
    loss_fn = lambda out, y: torch.nn.BCEWithLogitsLoss(reduction='sum')(out[:,:,0], y)

    losses = []
    
    for epoch in range(epochs):
        for qid in range(dataset.trdlr.shape[0] - 1):
            if label_transformation == 'debias':
                lv = correction_op.debias(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])
            elif label_transformation == 'em_prob':
                lv = pr1(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])
            else:
                raise Exception('not implemented')
            if bernoulli:
                lv = np.random.binomial(1, lv)
#             qgdata.update_labels([np.random.binomial(1, correction_op.debias(dataset.clicks[qid], 
            qgdata.update_labels([lv], 
                                 [qid])
    
                
        net.train()
        labels_mean_list = {}
        for (x, y, qids) in train_dl:
            net.opt.zero_grad()
            out = net(x)
            loss = loss_fn(out, y)
            losses.append(loss.data)
            loss.backward()
            net.opt.step()
            qgdata.update_predicted(out, qids)
        pr1 = correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr,
                                   dataset.group_ids)
        
        if log_epoch(epoch, epochs):
            if hasattr(dataset, 'vafm'):
                valid_ndcg = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),predict_dnn(net.eval())(dataset.vafm, dataset.vadlr)).NDCG(10)
            else:
                valid_ndcg = np.nan
            train_ndcg = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),qgdata.predicted).NDCG(10)

            if verbose:
                print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean())})
            with open(results_file, 'a+') as f:
                json.dump({
                    'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                           'train_size': dataset.trdlr.shape[0]-1, 'layers': layers, 'train_clicks': int(train_clicks),
                           'epoch': epoch+1, 'learning_rate': learning_rate, 'dropout': dropout, 
                           'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean()),
                           'correction': correction_method, 'label_transformation': label_transformation, 'bernoulli': bernoulli,
                           'correction_params': correction_op.get_params()
                          }, f)
                f.write('\n')
    return correction_op.get_params()
    

def train_model(jobid, 
                dataset_name, dataset, correction_method, loss_str,
                epochs, layers,  learning_rate, dropout, rseed, 
                bernoulli, label_transformation,
                results_file, verbose=False):
    set_seed(rseed)
    
    
    qgdata = QueryGroupedLTRData(dataset.trfm, dataset.trdlr)
    train_dl = DataLoader(qgdata, batch_size=1, shuffle=False)
#     train_dl = DataLoader(qgdata, batch_size=1, shuffle=True)

    net = DNN(dataset.trfm.shape[1], layers, dropout)
    if torch.cuda.is_available():
        net.cuda(get_torch_device())
    print(net)
    
    train_clicks = 0
    for c in dataset.clicks:
        train_clicks += c.sum()
    
    net.opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if 'oracle' in correction_method:
        param_path = correction_method.split('_',1)[1]
        correction_method = 'oracle'
        
        with open(param_path, 'rb') as f:
            correction_params = pickle.load(f)
    else:
        correction_params = rbem(jobid, dataset_name, dataset, correction_method,
                                 epochs, layers,  learning_rate, dropout, rseed, 
                                 results_file, bernoulli, label_transformation,
                                 verbose)
        
    correction_op = Correction(correction_method, 0.)
#     First we need to call this module to read the shapes
    correction_op.expmax(np.ones(dataset.trfm.shape[0])*0.5, 
                               dataset.clicks, dataset.sessions, dataset.trdlr, dataset.group_ids)
    correction_op.load_oracle_values(correction_params)
    
    loss_fn = lambda out, y: torch.nn.BCEWithLogitsLoss(reduction='sum')(out[:,:,0], y)
    if loss_str != 'BCE':
        loss_fn = eval(loss_str)
    losses = []
    
    results_file = os.path.join(results_file, jobid + '.json')

    for qid in range(dataset.trdlr.shape[0] - 1):
        lv = correction_op.debias(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])
        qgdata.update_labels([lv], [qid])
        
    max_valid_ndcg = -1
    for epoch in range(epochs):
        net.train()
        labels_mean_list = {}
        for (x, y, qids) in train_dl:
            net.opt.zero_grad()
            out = net(x)
            loss = loss_fn(out, y)
            losses.append(loss.data)
            loss.backward()
            net.opt.step()
            qgdata.update_predicted(out, qids)
        
        if log_epoch(epoch, epochs):
            train_ndcg = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),qgdata.predicted).NDCG(10)
            valid_ndcg = test_ndcg = -1
            
            if hasattr(dataset, 'vafm'):
                valid_ndcg = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),predict_dnn(net.eval())(dataset.vafm, dataset.vadlr)).NDCG(10)
            if hasattr(dataset, 'tefm'):
                test_ndcg = evaluate(dataset, predict_dnn(net.eval()), 
                                     saveto = None if max_valid_ndcg <= valid_ndcg else results_file.replace('.json','.out.pkl'))
            
            max_valid_ndcg = max(valid_ndcg, max_valid_ndcg)
            
            if verbose:
                print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 'test': test_ndcg, 'loss': float(np.array(losses).mean())})
            with open(results_file, 'a+') as f:
                json.dump({
                           'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                           'train_size': dataset.trdlr.shape[0]-1, 'layers': layers, 'train_clicks': int(train_clicks),
                           'epoch': epoch+1, 'learning_rate': learning_rate, 'dropout': dropout, 
                           'train': train_ndcg, 'valid': valid_ndcg, 'test':test_ndcg, 'loss': float(np.array(losses).mean()),
                           'correction': correction_method, 'correction_params': correction_op.get_params(),
                           'label_transformation': label_transformation, 'bernoulli': bernoulli,
                          }, f)
                f.write('\n')
    return net


def predict_dnn(net):
  
    if torch.cuda.is_available():
        torch_ = torch.cuda
    else:
        torch_ = torch
    def predict(fm, dlr):
        dl = DataLoader(QueryGroupedLTRData(fm, dlr), batch_size=1, shuffle=False)
        y = []
        with torch.no_grad():
          for (x, _, _) in dl:
            output = net(x)[0,:,:]
#             print(output.shape)
            y.append(np.mean(output.cpu().data.numpy(), 1))
        return np.concatenate(y)
    return predict

