import numpy as np
from en.enbase import EnsembleBase
from nn.wknkn import WKNKN

class EGS(EnsembleBase):
    """
    ensemble with global imbalance based sampling 
    """
    
    def __init__(self, base_model = WKNKN(k=7,T=0.8), n_models=5, seed=0, p=0.9, s=1e-1):
        super().__init__(base_model, n_models, seed)
        self.p = p # the ratio of sampling         
        self.s = s # the smoth parameter
        self.copyable_attrs.append('p')
        self.copyable_attrs.append('s')

    def _undersampling(self, intMat, prob_d, prob_t, seed):
        
        prng = np.random.RandomState(seed)
        nd, nt = intMat.shape
        nd_ret, nt_ret = int(nd*self.p), int(nt*self.p) 
        
        index_d_ret = prng.choice(nd, nd_ret, replace=False, p=prob_d) # the retained drug indices        
        index_t_ret = prng.choice(nt, nt_ret, replace=False, p=prob_t) # the retained target indices
        
        return index_d_ret, index_t_ret
    
    
    def _get_subset(self, intMat, drugMat, targetMat, index_d_ret, index_t_ret):
        Y = intMat[np.ix_(index_d_ret,index_t_ret)]
        D = drugMat[np.ix_(index_d_ret,index_d_ret)] 
        T = targetMat[np.ix_(index_t_ret,index_t_ret)]
        return Y, D, T
    
    def _cal_probs(self, intMat, drugMat, targetMat):
        s = self.s
        c1_ds = np.sum(intMat, axis=1) # the number of "1"s for each drug
        c1_ts = np.sum(intMat, axis=0) # the number of "1"s for each target
        c1_ds += s
        c1_ts += s
        prob_d = c1_ds/np.sum(c1_ds)
        prob_t = c1_ts/np.sum(c1_ts)
        return prob_d, prob_t
        
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self._initilize_model()
        self._index_map = [] 
        # compute probability to be selected
        prob_d, prob_t = self._cal_probs(intMat, drugMat, targetMat)
        for i in range(self.n_models):
            index_d_ret, index_t_ret = self._undersampling(intMat, prob_d, prob_t, self._seeds[i])
            self._index_map.append((index_d_ret, index_t_ret))
            Y, D, T = self._get_subset(intMat, drugMat, targetMat, index_d_ret, index_t_ret)
            self._models[i].fit(Y, D, T, self._cvs)
    
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        if self._cvs == 2:
            nm_t = np.zeros(self._n_targets, dtype=int) # the number of models for each target
            for i in range(self.n_models):
                index_d_ret, index_t_ret = self._index_map[i]
                D = drugMatTe[:,index_d_ret]
                T = targetMatTe[np.ix_(index_t_ret,index_t_ret)]
                scores1 = self._models[i].predict(D,T)
                scores[:,index_t_ret] += scores1
                nm_t[index_t_ret] += 1
            for t in range(self._n_targets):
                if nm_t[t] != 0: 
                    scores[:,t] /= nm_t[t]
        elif self._cvs == 3:
            nm_d = np.zeros(self._n_drugs, dtype=int) # the number of models for each target drug
            for i in range(self.n_models):
                index_d_ret, index_t_ret = self._index_map[i]
                D = drugMatTe[np.ix_(index_d_ret,index_d_ret)]
                T = targetMatTe[:,index_t_ret]    
                scores1 = self._models[i].predict(D,T)
                scores[index_d_ret,:] += scores1
                nm_d[index_d_ret] += 1
            for d in range(self._n_drugs):
                if nm_d[d] != 0: 
                    scores[d,:] /= nm_d[d]
        elif self._cvs == 4:
            for i in range(self.n_models):
                index_d_ret, index_t_ret = self._index_map[i]
                D = drugMatTe[:,index_d_ret]
                T = targetMatTe[:,index_t_ret]
                scores1 = self._models[i].predict(D,T)
                scores += scores1
            scores /= self.n_models
                
        return scores
