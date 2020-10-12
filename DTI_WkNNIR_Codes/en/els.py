import numpy as np
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
from en.egs import EGS
from nn.wknkn import WKNKN

class ELS(EGS):
    """
    ensemble with under sampling considering local imbalance
    """
    def __init__(self, base_model = WKNKN(k=7,T=0.8), n_models=5, seed=0, p=0.9, k=5, s=1e-1):
        super().__init__(base_model, n_models, seed, p, s)
        self.k = k # the k for calculate local imbalance
        self.copyable_attrs.append('k')

    def _undersampling(self, intMat, prob_d, prob_t, seed):
        """
        undersampling based on local imbalance:
            W = \sum C
        """
        prng = np.random.RandomState(seed)
        nd, nt = intMat.shape
        nd_ret, nt_ret = int(nd*self.p), int(nt*self.p)         
            
        index_d_ret = prng.choice(nd, nd_ret, replace=False, p=prob_d)        
        index_t_ret = prng.choice(nt, nt_ret, replace=False, p=prob_t)
        
        return index_d_ret, index_t_ret
    
    def _cal_probs(self, intMat, drugMat, targetMat):
        s = self.s #/self.k

        self._C_d = self._cal_limb(drugMat, intMat, self.k)
        self._C_t = self._cal_limb(targetMat, intMat.T, self.k)
        # C = C_d + C_t.T
        
        w_d = np.sum(self._C_d, axis=1)+s 
        w_t = np.sum(self._C_t, axis=1)+s
        
        prob_d = w_d/np.sum(w_d)
        prob_t = w_t/np.sum(w_t)        
        return prob_d, prob_t


    def _cal_limb(self, S, Y, k):
        """
        calulate and Local Imbalance of each interaction, 
        S is the similairty matrix
        Y is the interaction matrix
        k is the number of neighbours
        """
        if k > S.shape[0]:
            k = S.shape[0]
        
        S1 = S - np.diag(np.diag(S)) # ensure diagonal elements are 0
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S1.shape))
        knns = neigh.kneighbors(1 - S1, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            for j in range(Y.shape[1]):
                if Y[i,j] == 1: # only consider "1" 
                    C[i,j] = k-np.sum(Y[ii,j])
        C = C/k
        
        return C    