import numpy as np

import fairness.DTR as DTR
import fairness.EEL as EEL


def _get_edges(args):
    edges = np.zeros([len(args), len(args)])
    for i in range(len(args)):
        for j in range(i+1, len(args)):
            if args[i] > args[j]:
                edges[args[j]][args[i]] = 1
    return edges


def _insert_to_down(merged, PPG, i_u, up):
    Nu = PPG.shape[0]

    if i_u < up.shape[0] - 1:
        after_ind = int(np.where(merged == up[i_u + 1])[0])
    else:
        after_ind = merged.shape[0]

    if after_ind == i_u + 1:
        # print('no space to move')
        return

    for i_d in range(i_u+1, after_ind):
        if PPG[merged[i_u]][merged[i_d]] == 0:
            break
        q_u, q_d = 0, 0
        
        q = q_u + q_d - (q_u * q_d)

        q *= 1. - PPG[merged[i_u]][merged[i_d]]
       
        if q == 1:
            break
        sampling_prob = PPG[merged[i_u]][merged[i_d]] / (1. - q)
        if sampling_prob < 0 or sampling_prob > 1 or np.random.binomial(1, sampling_prob) == 0:
            break

    if i_d > i_u + 1:
        shift = merged[i_u+1:i_d]
        merged_i_u = merged[i_u]
        merged[i_u:i_d-1] = shift
        merged[i_d-1] = merged_i_u

    
def get_permutation(selected):
    return np.arange(selected.shape[0]) + selected.sum(1) - selected.sum(0)

def _PPG_merge(up, down, PPG):
    Nu = up.shape[0]
    Nd = down.shape[0]
    
    down += Nu
    merged = np.concatenate([up, down])

    for i_u in reversed(range(Nu)):
        _insert_to_down(merged, PPG, i_u, up)
    return merged

def _PPG_sample(PPG):
    n = PPG.shape[0]
    mid = n // 2
    if n == 1:
        return np.array([0])
    if n == 2:
        if np.random.binomial(1,PPG[0,1]):
            return np.array([1,0])
        return np.array([0,1])
    up = _PPG_sample(PPG[:mid,:][:,:mid])
    down = _PPG_sample(PPG[mid:,:][:,mid:])
    merged = _PPG_merge(up, down, PPG)
    return merged

def _PPG_sample_sessions(PPG, dlr):
    sampled = []
    for qid in range(dlr.shape[0]-1):
        s, e = dlr[qid:qid+2]
        sampled.append(_PPG_sample(PPG[s:e,:][:,s:e])+s)
    return np.concatenate(sampled)
            

class Learner:
    def __init__(self, PPG_mat, samples_cnt, objective_ins, sorted_docs, dlr, intra, inter) -> None:
        self.ref_permutation = sorted_docs
        self.dlr = dlr
        self.objective = objective_ins
        self.n = len(self.ref_permutation)
        self.samples_cnt = samples_cnt
        if PPG_mat is None:
            PPG_mat = 0.5 * np.triu(np.ones((self.n,self.n)), 1)

        for i in range(self.n):
            for j in range(i+1, self.n):
                if intra[self.ref_permutation[i]] == intra[self.ref_permutation[j]] \
                    or inter[self.ref_permutation[i]] != inter[self.ref_permutation[j]]:
                    PPG_mat[i,j] = 0
        self.PPG = PPG_mat

        self.intra = intra
        self.inter = inter


    def _update_ref(self, new_ref):
        edges = _get_edges(new_ref)

        # print('edges:', edges)

        # print('before inversion:', self.PPG)
        
        self.PPG -= edges
        self.PPG = np.abs(self.PPG)

        self.PPG = self.PPG[new_ref,:][:,new_ref]
        self.PPG += self.PPG.T
        self.PPG *= np.triu(np.ones((self.n,self.n)), 1)
        

        self.ref_permutation = self.ref_permutation[new_ref]


    def fit(self, epochs, lr, verbose=0):
        self.verbose = verbose

        min_f = np.inf
        min_b = np.arange(self.n)

        if self.verbose > 0:
            print(self.ref_permutation, 'inter:', self.inter, 'intra:', self.intra)

        min_changed_epoch = -1
        for epoch in range(epochs):
            grad = np.zeros_like(self.PPG)
            safe_PPG = np.copy(self.PPG)
            safe_PPG[self.PPG == 0] = -1
            inv = 1./safe_PPG
            inv[inv<0] = 0
            safe_PPG[self.PPG == 0] = 2
            inv2 = 1./(1. - safe_PPG)
            inv2[inv2<0] = 0
            fs = 0
            min_changed = False
            for s in range(self.samples_cnt):
                b = _PPG_sample_sessions(self.PPG, self.dlr)
                if self.verbose > 1:
                    print(b, '->', self.ref_permutation[b])
                f = self.objective.eval(self.ref_permutation[b])
                if f < min_f:
                    min_b = b
                    min_f = f
                    min_changed = True
                    min_changed_epoch = epoch
                e = _get_edges(b)
                fs += f
                
                grad += f * ((e*inv) - ((1.-e)*inv2))
            grad /= self.samples_cnt
            self.PPG -= lr * grad
            if self.verbose > 0:
                print('min_f:', min_f, ', mean_f:', fs/self.samples_cnt) #, min_b])
            self.PPG[self.PPG < 0] = 0.05
            self.PPG[self.PPG >= 0.95] = 0.95

            if min_changed:
                self._update_ref(min_b)
                min_b = np.arange(self.n)
                if self.verbose > 0:
                    print('new ref permutation:\n',self.ref_permutation)
                    print('intra:\n', self.intra[self.ref_permutation])
                if self.verbose > 2:
                    print(self.PPG)

            if epoch - min_changed_epoch > 10:
                break

        return self.ref_permutation[min_b]

    
def learn_one_PPG(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]
    y_pred_s, g_s, sorted_docs_s, dlr_s = \
        EEL.copy_sessions(y=y_pred[s:e], g=g[s:e], sorted_docs=y_pred[s:e].argsort()[::-1], sessions=sessions_cnt)
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure)
        
    learner = Learner(  PPG_mat=None, samples_cnt=samples_cnt, 
                                objective_ins=objective_ins, 
                                sorted_docs = sorted_docs_s, 
                                dlr = dlr_s,
                                intra = g_s,
                                inter = np.repeat(dlr_s[:-1], np.diff(dlr_s)))
    vals = learner.fit(epochs, lr, verbose=verbose)
    return vals

def learn_all_PPG(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
    
    for qid in range(dlr.shape[0] - 1):
        min_b = learn_one_PPG(metric, qid, 0, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        
    return sorted_docs

