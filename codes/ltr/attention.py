
import pickle
import time
import os
import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd



# from correction import *
from ltr.allrank.model import *
from ltr.allrank.lambdaLoss import lambdaLoss as lambdaLoss
import ltr.metrics as metrics


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

class LTRData_correction(Dataset):
    def __init__(self, fm, dlr):
        
        self.fm_by_qid = np.split(fm, dlr[1:-1])
        self.predicted = [np.ones(dlr[qid+1] - dlr[qid])*0.5 for qid in range(dlr.shape[0] - 1)]
        self.lv = [None for _ in range(dlr.shape[0] - 1)]
        self.dev = get_torch_device()
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
            
    def update_labels(self, labels, qids):
        for qid, label in zip(qids, labels):
            self.lv[qid] = label[:]
        
    def update_predicted(self, ys, qids):
        for qid, y in zip(qids, ys):
            self.predicted[qid] = y.cpu().data[:self.lv[qid].shape[0]]
            
    def __len__(self):
        return len(self.fm_by_qid)
    
    def __getitem__(self, qid):
        feature = self.torch_.FloatTensor(self.fm_by_qid[qid], device=self.dev)
        lv = self.torch_.FloatTensor(self.lv[qid], device=self.dev) if self.lv[qid] is not None else None
        return feature, lv, qid
    

    
class LTRData(Dataset):
    def __init__(self, fm, lv, dlr):
        self.fm = np.split(fm, dlr[1:-1])
        self.lv = np.split(lv, dlr[1:-1]) if lv is not None else [np.zeros(dlr[qid+1] - dlr[qid]) for qid in range(dlr.shape[0] - 1)]
        self.predicted = [np.ones(dlr[qid+1] - dlr[qid])*0.5 for qid in range(dlr.shape[0] - 1)]
        self.dev = get_torch_device()
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
            
    def update_predicted(self, ys, qids):
        for qid, y in zip(qids, ys):
            self.predicted[qid] = y.cpu().data[:self.lv[qid].shape[0]]
            
    def __len__(self):
        return len(self.fm)
    
    def __getitem__(self, qid):
        feature = self.torch_.FloatTensor(self.fm[qid], device=self.dev)
        lv = self.torch_.FloatTensor(self.lv[qid], device=self.dev)
        return feature, lv, qid
    
def collate_LTR(batch):
    batch_lens = [feature.shape[0] for feature, lv, qid in batch]
    max_len = max(batch_lens)
    X = torch.stack([torch.nn.functional.pad(feature,pad=(0,0,0,max_len-feature.shape[0])) for feature, lv, qid in batch])
    Y = torch.stack([torch.nn.functional.pad(lv,pad=(0,max_len-lv.shape[0]), value=-1) for feature, lv, qid in batch]) if batch[0][1] is not None else None
    qids = [qid for feature, lv, qid in batch]
    indices = torch.stack([torch.LongTensor(np.pad(np.arange(0, sample_size), (0, max_len - sample_size), "constant", constant_values=-1)) for sample_size in batch_lens], dim=0)
    return X, Y, indices, qids
    

def set_seed(random_seed):
    import random
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def predict_correction(net, fm, dlr, lv):
    qgdata = LTRData_correction(fm, dlr)
    valid_dl = DataLoader(qgdata, batch_size=1, shuffle=False, collate_fn = collate_LTR)
    net.eval()
    
    for qid in range(dlr.shape[0] - 1):
        qgdata.update_labels([lv[dlr[qid]:dlr[qid]+1]], [qid])
            
    preds = []
    with torch.no_grad():
        for (x, y, indices, _) in valid_dl:
            mask = (y == -1)
            output = net(x, mask, indices).squeeze(dim=0)
            preds.append(output.cpu().data.numpy())
    return np.concatenate(preds)
    

def predict(net, fm, dlr):
    qgdata = LTRData(fm, None, dlr)
    valid_dl = DataLoader(qgdata, batch_size=1, shuffle=False, collate_fn = collate_LTR)
    net.eval()
    
    preds = []
    with torch.no_grad():
        for (x, y, indices, _) in valid_dl:
            mask = (y == -1)
            output = net(x, mask, indices).squeeze(dim=0)
            preds.append(output.cpu().data.numpy())
    return np.concatenate(preds)
    

def cltr_correction(jobid, dataset_name, correction_method,
         net_config, dataset,
         epochs, learning_rate, rseed, 
         bernoulli, 
         is_rbem,
         results_file, verbose=False):
    
    if 'oracle' in correction_method:
        is_rbem = False
    if 'outlier' not in correction_method:
        outlierness = np.zeros_like(dataset.trfm[:,0])
        dataset.group_ids, dataset.biglist_index = outlier2group(outlierness, dataset.sessions, dataset.trdlr)
    
    set_seed(rseed)
    
    
    qgdata = LTRData_correction(dataset.trfm, dataset.trdlr)
    train_dl = DataLoader(qgdata, batch_size=1, shuffle=True, collate_fn = collate_LTR)

    net = make_model(**net_config['model'], n_features=dataset.trfm.shape[1])
    if torch.cuda.is_available():
        net.cuda(get_torch_device())
    print(net)
    
    train_clicks = 0
    for c in dataset.clicks:
        train_clicks += c.sum()
    
    net.opt = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    
    correction_params = None
    if not is_rbem:
        if 'oracle' in correction_method:
            param_path = correction_method.split('_',1)[1]
            correction_method = 'oracle'

            with open(param_path, 'rb') as f:
                correction_params = pickle.load(f)
        else:
            correction_params = cltr(jobid, dataset_name, correction_method,
                                     net_config, dataset,
                                     epochs, learning_rate, rseed, 
                                     bernoulli, 
                                     is_rbem = True,
                                     results_file = results_file, verbose = verbose)
       
        correction_op = Correction(correction_method.replace('outlier_', ''), 0.)
    #     First we need to call this module to read the shapes
        correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr, dataset.group_ids)
        correction_op.load_oracle_values(correction_params)
        
        pr1 = None
        loss_fn = lambdaLoss
        results_file = os.path.join(results_file, jobid + '.json')
        
    else:
        correction_op = Correction(correction_method.replace('outlier_', ''), 0.2)

        pr1 = correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr, dataset.group_ids)

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        results_file = os.path.join(results_file, jobid + '_rbem.json')
    
    
    
    losses = []
    
    if pr1 is None:
        for qid in range(dataset.trdlr.shape[0] - 1):
            lv = correction_op.debias(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])
            qgdata.update_labels([lv], [qid])
    
    for epoch in range(epochs):
        if pr1 is not None:
            for qid in range(dataset.trdlr.shape[0] - 1):
                lv = pr1(dataset.clicks[qid], dataset.sessions[qid], dataset.biglist_index[qid])

                if bernoulli:
                    lv = np.random.binomial(1, lv)

                qgdata.update_labels([lv], [qid])
    
        net.train()
        for (x, y, indices, qids) in train_dl:
            net.opt.zero_grad()
            mask = (y == -1)
            out = net(x, mask, indices)
            out[mask] = -1e6
            y[mask] = 0
            loss = loss_fn(out, y)
            losses.append(loss.data)
            loss.backward()
            net.opt.step()
            qgdata.update_predicted(out, qids)
        
        if pr1 is not None:
            pr1 = correction_op.expmax(qgdata.predicted, dataset.clicks, dataset.sessions, dataset.trdlr,
                                       dataset.group_ids)
        
        if log_epoch(epoch, epochs):
            train_ndcg = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),np.concatenate(qgdata.predicted, 0)).NDCG(10)
            valid_ndcg = -1
            if hasattr(dataset, 'vafm'):
                valid_ndcg = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),predict_correction(net,dataset.vafm, dataset.vadlr, dataset.valv)).NDCG(10)
            
            if verbose:
                print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean())})
            with open(results_file, 'a+') as f:
                json.dump({
                    'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                           'train_size': dataset.trdlr.shape[0]-1, 'train_clicks': int(train_clicks),
                           'epoch': epoch+1, 'learning_rate': learning_rate, 
                           'config': net_config,
                           'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean()),
                           'correction': correction_method, 'bernoulli': bernoulli,
                           'correction_params': correction_op.get_params()
                          }, f)
                f.write('\n')
         
    if hasattr(dataset, 'tefm'):
        test_ndcg = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),predict_correction(net,dataset.tefm, dataset.tedlr, dataset.telv)).NDCG(10)

        with open(results_file, 'a+') as f:
            json.dump({
                'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                       'train_size': dataset.trdlr.shape[0]-1, 'train_clicks': int(train_clicks),
                       'epoch': epoch+1, 'learning_rate': learning_rate,
                       'config': net_config,
                       'train': train_ndcg, 'valid': valid_ndcg, 'test': test_ndcg, 'loss': float(np.array(losses).mean()),
                       'correction': correction_method, 'bernoulli': bernoulli,
                       'correction_params': correction_op.get_params()
                      }, f)
            f.write('\n')
    
    if is_rbem:
        return correction_op.get_params()
    else:
        return net


    
def cltr(cfg, dataset, groups, learn_group = False, verbose = True):
    jobid = cfg['jobid']
    dataset_name = cfg['ltr']['dataset']
    net_config = cfg['ltr']['net_config']
    epochs = cfg['ltr']['epochs']
    learning_rate = cfg['ltr']['lr']
    beta = cfg['group']['beta']
    results_file = cfg['ltr']['results_file']
        
    set_seed(cfg['ltr']['seed'])
    
    if verbose:
        print('train ratio:', groups['trg'].mean())
        print('valid ratio:', groups['vag'].mean())
        
    if cfg.group.affected == 1:
        groups['trg'] = 1 - groups['trg']
        groups['vag'] = 1 - groups['vag']
        groups['teg'] = 1 - groups['teg']
        
    tr_beta = groups['trg'].astype(np.float)
    va_beta = groups['vag'].astype(np.float)
    
    if learn_group:
        dataset.trlv = tr_beta
        dataset.valv = va_beta
        dataset.telv = groups['teg'].astype(np.float)
        dataset.trfm[:, cfg.group.feature_id] = 0
        dataset.vafm[:, cfg.group.feature_id] = 0
        dataset.tefm[:, cfg.group.feature_id] = 0
        
        train_ndcg_b = np.nan
    else:
        dataset.trlv = dataset.trlv.astype(np.float)
        dataset.valv = dataset.valv.astype(np.float)
        
        

        tr_beta[tr_beta == 0] = beta
        va_beta[va_beta == 0] = beta
        if verbose:
            print('train group bias mean:', tr_beta.mean())
            print('valid group bias mean:', va_beta.mean())
            
        train_ndcg_b = metrics.LTRMetrics(np.round(dataset.trlv/4,0), np.diff(dataset.trdlr), dataset.trlv * tr_beta).NDCG(10)
        
        dataset.trlv *= tr_beta
        dataset.valv *= va_beta
        
    qgdata = LTRData(dataset.trfm, dataset.trlv, dataset.trdlr)
    train_dl = DataLoader(qgdata, batch_size=1, shuffle=True, collate_fn = collate_LTR)

    net = make_model(**net_config['model'], n_features=dataset.trfm.shape[1])
    if torch.cuda.is_available():
        net.cuda(get_torch_device())
#     print(net)
    
    net.opt = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    

    loss_fn = lambdaLoss
    results_file = os.path.join(results_file, f'{jobid}.json')
    
    
    losses = []
    predicted = []
    
    for epoch in range(epochs):
        net.train()
        for (x, y, indices, qids) in train_dl:
            net.opt.zero_grad()
            mask = (y == -1)
            out = net(x, mask, indices)
            out[mask] = -1e6
            y[mask] = 0
            loss = loss_fn(out, y)
            losses.append(loss.data)
            loss.backward()
            net.opt.step()
            qgdata.update_predicted(out, qids)
        
        if log_epoch(epoch, epochs):
            train_ndcg = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),np.concatenate(qgdata.predicted, 0)).NDCG(10)
            valid_ndcg = -1
            if hasattr(dataset, 'vafm'):
                valid_ndcg = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),predict(net,dataset.vafm, dataset.vadlr)).NDCG(10)
            
            if verbose:
                print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean())})
            with open(results_file, 'a+') as f:
                json.dump({
                    'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                           'train_size': dataset.trdlr.shape[0]-1, 
                           'epoch': epoch+1, 
                           'config': dict(cfg.group),
                           'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean()),
                          }, f)
                f.write('\n')
         
    if hasattr(dataset, 'tefm'):
        y_pred = predict(net,dataset.tefm, dataset.tedlr)
        
        with open(results_file.replace('.json','.out.pkl'), 'wb') as f:
            pickle.dump({'dlr':dataset.tedlr,
                        'lv': dataset.telv,
                        'pred': y_pred,
                        'g': groups['teg'],
                        'group_config': dict(cfg.group),
                        'dataset': dataset_name,
                        'file_name': results_file}, 
                        f)
            
        
        test_ndcg = metrics.LTRMetrics(dataset.telv, np.diff(dataset.tedlr), y_pred).NDCG(10)
        
        if not learn_group:
            g_lv = dataset.telv[:] + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 0] = 0
            g_pred[groups['teg'] == 0] = 0
            test_ndcg_1 = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
            g_lv = dataset.telv[:] + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 1] = 0
            g_pred[groups['teg'] == 1] = 0
            test_ndcg_0 = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
            g_lv = dataset.telv[:] + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 0] = 0
            g_pred[groups['teg'] == 0] = 0
            test_ndcg_1_sanity = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
        else:
            test_ndcg_0 = test_ndcg_1 = test_ndcg_1_sanity = np.nan
            
            
        
        test_ndcg_b = metrics.LTRMetrics(np.round(dataset.telv/4,0), np.diff(dataset.tedlr), y_pred).NDCG(10)
        
        if not learn_group:
            g_lv = np.round(dataset.telv[:]/4,0) + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 0] = 0
            g_pred[groups['teg'] == 0] = 0
            test_ndcg_1_b = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
            g_lv = np.round(dataset.telv[:]/4,0) + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 1] = 0
            g_pred[groups['teg'] == 1] = 0
            test_ndcg_0_b = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
            g_lv = np.round(dataset.telv[:]/4,0) + 0.
            g_pred = y_pred[:] + 0.
            g_lv[groups['teg'] == 0] = 0
            g_pred[groups['teg'] == 0] = 0
            test_ndcg_1_sanity_b = metrics.LTRMetrics(g_lv, np.diff(dataset.tedlr), g_pred).NDCG(10)
        else:
            test_ndcg_0_b = test_ndcg_1_b = test_ndcg_1_sanity_b = np.nan

            
        if verbose:
            print({'epoch': epoch+1, 'train': train_ndcg, 'valid': valid_ndcg, 
                   'test': test_ndcg, 'test_0':test_ndcg_0, 'test_1':test_ndcg_1, 
                   'loss': float(np.array(losses).mean())})
        with open(results_file, 'a+') as f:
            json.dump({
                'jobid': jobid, 'dataset': dataset_name, 'train_docs': dataset.trfm.shape[0], 
                       'train_size': dataset.trdlr.shape[0]-1,
                       'epoch': epoch+1,
                       'config': dict(cfg.group),
                       'train': train_ndcg, 'valid': valid_ndcg, 'loss': float(np.array(losses).mean()),
                       'train_ndcg_b': train_ndcg_b,
                       'test': test_ndcg, 'test_0':test_ndcg_0, 'test_1':test_ndcg_1, 'test_1_sanity':test_ndcg_1_sanity, 
                       'test_b': test_ndcg_b, 'test_0_b':test_ndcg_0_b, 'test_1_b':test_ndcg_1_b, 'test_1_sanity_b':test_ndcg_1_sanity_b, 
                      }, f)
            f.write('\n')
    
    return net

