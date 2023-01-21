import json
import pickle
import os
import numpy as np
import pandas as pd
import hydra

from fairness.PPG import learn_all_PPG
import fairness.DTR as DTR
import fairness.EEL as EEL


def evaluate_one(metric, qid, lv, g, dlr, output_permutation, exposure, sessions_cnt):
    s, e = dlr[qid:qid+2]
    permutation = output_permutation[qid]
    lv_s, g_s, sorted_docs_s, dlr_s = \
        EEL.copy_sessions(y=lv[s:e], g=g[s:e], sorted_docs=lv[s:e].argsort()[::-1], sessions=sessions_cnt)
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = lv_s, g = g_s, dlr = dlr_s, exposure=exposure, grade_levels = 2)
    else:
        objective_ins = DTR.DTR(y_pred = lv_s, g = g_s, dlr = dlr_s, exposure=exposure)
        
    
    osl = e - s
    argsort = lv[s:e].argsort()[::-1]
    idcg = ((2.**lv[s:e][argsort][:min(osl,10)] - 1.) / (np.log2(2+np.arange(min(osl,10))))).sum()
    ndcg = 0
    if idcg > 0:
        for i in range(sessions_cnt):
            ndcg += ((2.**lv[s:e][permutation[i*osl:(i+1)*osl]-(i*osl)][:min(osl,10)] - 1.) / (np.log2(2+np.arange(min(osl,10))))).sum() / idcg
        
    return objective_ins.eval(permutation), ndcg / sessions_cnt
 
def evaluate_all(metric, valids, lv, g, dlr, output_permutation, exposure, sessions_cnt):
    eel_res, eer_res, eed_res, ndcgs = [], [], [], []
#     for qid in range(dlr.shape[0] - 1):
    for qid in valids:
        s,e = dlr[qid:qid+2]
#         if len(np.unique(g[s:e])) == 1:
#             continue
        out1, ndcg = evaluate_one(metric, qid, lv, g, dlr, output_permutation, exposure, sessions_cnt)
        eel = out1
        eel_res.append(eel)
        ndcgs.append(ndcg)
    return np.array(eel_res), np.array(ndcgs)

def cut_topk(utility, topk):
    g = utility.g.astype(np.float)
    pred = utility.pred
    lv = utility.lv
    dlr = utility.dlr
    pred_k, lv_k, g_k, dlr_k = [], [], [], [0]
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        args = np.argsort(pred[s_i:e_i])[::-1]
        args = args[:topk]
        lv_k.append(lv[s_i:e_i][args])
        g_k.append(g[s_i:e_i][args])
        pred_k.append(pred[s_i:e_i][args])
        dlr_k.append(args.shape[0])
    utility.lv = np.concatenate(lv_k, 0)
    utility.g = np.concatenate(g_k, 0)
    utility.pred = np.concatenate(pred_k, 0)
    utility.dlr = np.cumsum(dlr_k)
    
def remove_single_groups(utility):
    g = utility.g.astype(np.float)
    pred = utility.pred
    lv = utility.lv
    dlr = utility.dlr
    pred_k, lv_k, g_k, dlr_k = [], [], [], [0]
    for qid in range(dlr.shape[0] - 1):
        s_i, e_i = dlr[qid:qid+2]
        g_ = g[s_i:e_i]
        if g_.min() == g_.max():
            continue
        lv_k.append(lv[s_i:e_i])
        g_k.append(g[s_i:e_i])
        pred_k.append(pred[s_i:e_i])
        dlr_k.append(e_i-s_i)
    utility.lv = np.concatenate(lv_k, 0)
    utility.g = np.concatenate(g_k, 0)
    utility.pred = np.concatenate(pred_k, 0)
    utility.dlr = np.cumsum(dlr_k)
        
        
    
@hydra.main(version_base=None, config_path="configs", config_name="fairness")
def my_app(cfg):
    
    with open(os.path.join(cfg.fairness.utility_path, cfg.fairness.utility_name), 'rb') as f:
        utility = type('utility', (object, ), pickle.load(f))
        
    cut_topk(utility, cfg.fairness.topk)
    remove_single_groups(utility)
    
    y_pred = utility.pred
    if cfg.fairness.utility_type == 'labels':
        beta = utility.group_config['beta']
        te_beta = utility.g.astype(np.float)
        te_beta[te_beta == 0] = beta
        y_pred = utility.lv * te_beta
        
    if cfg.fairness.exposure == 'log':
        exposure = np.array([1./np.log2(2+i) for i in range(1,np.diff(utility.dlr).max()+2)])
    else:
        raise 'exposure type not supported!'
    
    if cfg.fairness.learn:
        output_permutation = learn_all_PPG(metric = cfg.fairness.metric, y_pred = y_pred, g = utility.g, dlr = utility.dlr, 
                                           epochs = cfg.fairness.epochs, lr = cfg.fairness.lr, 
                                           exposure = exposure, grade_levels = cfg.fairness.grade_levels, 
                                           samples_cnt = cfg.fairness.samples, sessions_cnt = cfg.fairness.sessions)
        sessions_cnt = cfg.fairness.sessions
    else:
        output_permutation = [np.argsort(y_pred[utility.dlr[qid]:utility.dlr[qid+1]])[::-1] for qid in range(utility.dlr.shape[0] - 1)]
        sessions_cnt = 1
    
    fairs, ndcgs = evaluate_all(metric = cfg.fairness.metric, valids = np.arange(utility.dlr.shape[0] - 1), 
                                lv = utility.lv, g = utility.g, dlr = utility.dlr, 
                                output_permutation = output_permutation, exposure = exposure, 
                                sessions_cnt = sessions_cnt)
    print(np.array(fairs).mean())
    print(np.array(ndcgs).mean())
    
    with open(os.path.join(cfg.fairness.results_file, f'{cfg.jobid}.json'), 'w') as f:
        json.dump({'jobid': cfg.jobid,
                   'config': dict(cfg.fairness),
                   'queries': int(utility.dlr.shape[0] - 1),
                   'group_config': utility.group_config,
                   'dataset': utility.dataset,
                   'ratio': utility.g.astype(np.float).mean(),
                   'ndcg': np.array(ndcgs).mean(),
                   f'{cfg.fairness.metric}': np.array(fairs).mean()
                  },
                  f)
if __name__ == "__main__":
    my_app()
