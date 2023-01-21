import json
import pickle
import os
import numpy as np
import pandas as pd
import hydra

from ltr.attention import *


    
def read_datatset(datasets_info, name):
    with open(datasets_info[name]['path'], 'rb') as f:
        dataset = pickle.load(f)
    with open(datasets_info[name]['path'].replace('.pkl', datasets_info[name]['group_suffix']), 'rb') as f:
        groups = pickle.load(f)
    return type('ltr', (object, ), dataset), groups
        
def get_grouping(groups, feature_id, suffix, dataset):
    g = groups['grouping'][feature_id]
    return dict([(f'{split}g', g[f'{split}_{suffix}'][:getattr(dataset,f'{split}dlr')[-1]]) for split in ['tr', 'va', 'te']])


def normalize(fm, dlr):
    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        m = fm[s:e,:].mean(axis=0)
        z = fm[s:e,:].std(axis=0) + 1e-10
        fm[s:e, :] = (fm[s:e, :] - m) / z


    
@hydra.main(version_base=None, config_path="configs", config_name="ltr")
def my_app(cfg):
    with open(cfg.ltr.datasets_info, 'r') as f:
        datasets_info = json.load(f)

    dataset, all_groups = read_datatset(datasets_info, cfg.ltr.dataset)
    if cfg.ltr.toysize > 0:
        dataset.trdlr = dataset.trdlr[:cfg.ltr.toysize]
        dataset.trfm = dataset.trfm[:dataset.trdlr[-1], :]
        dataset.trlv = dataset.trlv[:dataset.trdlr[-1]]


        dataset.vadlr = dataset.vadlr[:cfg.ltr.toysize]
        dataset.vafm = dataset.vafm[:dataset.vadlr[-1], :]
        dataset.valv = dataset.valv[:dataset.vadlr[-1]]
    
    normalize(dataset.trfm, dataset.trdlr)
    normalize(dataset.tefm, dataset.tedlr)
    normalize(dataset.vafm, dataset.vadlr)

    groups = get_grouping(all_groups, cfg.group.feature_id, cfg.group.suffix, dataset)
    net = cltr(cfg, dataset = dataset, groups = groups, learn_group = cfg.ltr.learn_group, verbose = cfg.verbose)
    

if __name__ == "__main__":
    my_app()
