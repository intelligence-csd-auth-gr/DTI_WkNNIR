from base.inbase import InductiveModelBase
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
import numpy as np

class WKNKN(InductiveModelBase):
    """ 
    Implementation of WkNN
    """
    def __init__(self, k=5, T=0.7):
        self.k = k
        self.T = T # η
        
        self.copyable_attrs = ['k','T']
    
    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        
        if self._cvs == 2 and self.k > self._n_drugs:
            self.k = self._n_drugs
        if self._cvs == 3 and self.k > self._n_targets:
            self.k = self._n_targets
        if self._cvs == 4 and self.k > min(self._n_drugs, self._n_targets):
            self.k = min(self._n_drugs, self._n_targets)
            
        self._intMat=intMat
        
        if self._cvs == 2 or self._cvs == 4:
            self._neigh_d = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))   #(1- (drugMat - np.diag(np.diag(drugMat))))
        if self._cvs == 3 or self._cvs == 4:
            self._neigh_t = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_t.fit(np.zeros((self._n_targets,self._n_targets))) #(1- (targetMat - np.diag(np.diag(targetMat))))
        
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)

        if self._cvs == 2:  # new drug, known target
            ii_d = self._neigh_d.kneighbors(1 - drugMatTe, return_distance=False)
            for d in range(self._n_drugs_te): # d: each test drug
                ii = ii_d[d]
                for i in range(len(ii)):
                    w = self.T**i*drugMatTe[d,ii[i]]
                    scores[d, :] += w*self._intMat[ii[i], :]
                z = np.sum(drugMatTe[d,ii])
                scores[d,:]/=z
        elif self._cvs == 3: # known drug, new target
            jj_t = self._neigh_t.kneighbors(1 - targetMatTe, return_distance=False)
            for t in range(self._n_targets_te):
                jj = jj_t[t]
                for j in range(len(jj)):
                    w = self.T**j*targetMatTe[t,jj[j]]
                    scores[:,t] += w*self._intMat[:,jj[j]]
                z = np.sum(targetMatTe[t,jj])
                scores[:,t]/=z
        elif self._cvs == 4:
            scores = self._predict_cvs4_1_2(drugMatTe, targetMatTe)
                     
        return scores
       

    def _predict_cvs4_1_2(self, drugMatTe, targetMatTe):
        # the weight of i-th kNN(d) and j-th kNN(t) w = self.T**(i-1) * self.T**(j-1)
        # This method is faster but worse than _predict_cvs4_1 for nr, gpcr but better for ic and e
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        ii_d = self._neigh_d.kneighbors(1 - drugMatTe, return_distance=False)
        jj_t = self._neigh_t.kneighbors(1 - targetMatTe, return_distance=False)
        W = np.ones((self.k,self.k))
        for i in range(self.k):
            W[i,:] *= self.T**i
        for j in range(self.k):
            W[:,j] *= self.T**j
        
        for d in range(self._n_drugs_te):
            ii = ii_d[d]
            for t in range(self._n_targets_te):
                jj = jj_t[t]
                S = np.outer(drugMatTe[d,ii],targetMatTe[t,jj])
                Y = W*S*self._intMat[np.ix_(ii,jj)]
                scores[d,t]= np.sum(Y)/np.sum(S)  # z = np.sum(S)        
        return scores
        
        
    def __str__(self):
        return "Model: KNN: kd={:d}, kt={:d}".format(self.kd,self.kt) 