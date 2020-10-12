from base.inbase import InductiveModelBase
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
import numpy as np

# This is the method used in the paper
class WKNKNRI(InductiveModelBase):
    """
    Implementation of WkNNIR
    """
    def __init__(self, k=5, kr=5, T=0.7):
        self.k = k
        self.kr = kr # the k for recovery, which is same with kr for simplicity
        self.T = T # η

        self.copyable_attrs = ['k','kr','T']
    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self.kr = self.k # ensure self.kr = self.k 
        
        if self._cvs == 2: 
            if self.k > self._n_drugs:
                self.k = self._n_drugs
            if self.kr > self._n_targets:
                self.kr = self._n_targets
        if self._cvs == 3:
            if self.k > self._n_targets:
                self.k = self._n_targets
            if self.kr > self._n_drugs:
                self.kr = self._n_drugs

        if self._cvs == 4 and self.k > min(self._n_drugs, self._n_targets):
            self.k = min(self._n_drugs, self._n_targets)
            
        self._intMat=intMat
        self._drugMat = drugMat - np.diag(np.diag(drugMat))
        self._targetMat = targetMat - np.diag(np.diag(targetMat))
    
                        
        
        if self._cvs == 2:
            self._neigh_d = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            self._neigh_t = NearestNeighbors(n_neighbors=self.kr, metric='precomputed')
            self._neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            self._knns_t = self._neigh_t.kneighbors(1 - self._targetMat, return_distance=False)   
            
            Yt = self._recover_intMat_target()
            self._intMat = np.maximum(Yt, self._intMat)
            
        if self._cvs == 3:
            self._neigh_t = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            self._neigh_d = NearestNeighbors(n_neighbors=self.kr, metric='precomputed')
            self._neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            self._knns_d = self._neigh_d.kneighbors(1 - self._drugMat, return_distance=False)
                
            Yd = self._recover_intMat_drug()
            self._intMat = np.maximum(Yd, self._intMat)
            
        if self._cvs == 4:
            self._neigh_d = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            self._neigh_t = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
            self._neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            self._knns_d = self._neigh_d.kneighbors(1 - self._drugMat, return_distance=False)
            self._knns_t = self._neigh_t.kneighbors(1 - self._targetMat, return_distance=False)   

            limb_d = self._cal_limb(self._knns_d, self._intMat, self.k)
            limb_t = self._cal_limb(self._knns_t, self._intMat.T, self.k)
            r = limb_t/limb_d
            self._ri, self._rj = 1.0, 1.0
            if r<=1: # ri<=rj
                self._rj = r      
            else: # ri>rj
                self._ri = 1/r
            Yt = self._recover_intMat_target()*(1-limb_t)
            Yd = self._recover_intMat_drug()*(1-limb_d)
            Ydt = (Yt+Yd)/2
            self._intMat = np.maximum(Ydt, self._intMat) 
            
    def _recover_intMat_target(self):
        # recovery intMat by targetMat
        Yt = np.zeros(self._intMat.shape)
        jj_t = self._knns_t
        for t in range(self._n_targets):
            jj = jj_t[t]
            for j in range(len(jj)):
                w = self.T**j*self._targetMat[t,jj[j]]
                Yt[:,t] += w*self._intMat[:,jj[j]]
            z = np.sum(self._targetMat[t,jj])
            Yt[:,t]/=z
        # Yt[Yt>1] = 1
        return Yt
        
    
    def _recover_intMat_drug(self):
        # recovery intMat by drugMat
        Yd = np.zeros(self._intMat.shape)
        ii_d = self._knns_d
        for d in range(self._n_drugs): 
            ii = ii_d[d]
            for i in range(len(ii)):
                w = self.T**i*self._drugMat[d,ii[i]]
                Yd[d, :] += w*self._intMat[ii[i], :]
            z = np.sum(self._drugMat[d,ii])
            Yd[d,:]/=z
        # Yd[Yd>1] = 1
        return Yd
    
    
    
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
            scores = self._predict_cvs4_1_22(drugMatTe, targetMatTe)
                                 
        return scores
    

    def _predict_cvs4_1_22(self, drugMatTe, targetMatTe):
        # the weight of i-th kNN(d) and j-th kNN(t) w = self.T**(i-1) * self.T**(j-1)
        # besides, add local imbalance based coefficient ri and rj to i and j
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
       
        ii_d = self._neigh_d.kneighbors(1 - drugMatTe, return_distance=False)
        jj_t = self._neigh_t.kneighbors(1 - targetMatTe, return_distance=False)
        W = np.ones((self.k,self.k))
        for i in range(self.k):
            W[i,:] *= self.T**(i/self._ri) #self.T**(i/ri)
        for j in range(self.k):
            W[:,j] *= self.T**(j/self._rj)  # self.T**(j/rj)
        
        for d in range(self._n_drugs_te):
            ii = ii_d[d]
            for t in range(self._n_targets_te):
                jj = jj_t[t]
                S = np.outer(drugMatTe[d,ii],targetMatTe[t,jj])
                Y = W*S*self._intMat[np.ix_(ii,jj)]
                scores[d,t]= np.sum(Y)/np.sum(S)  # z = np.sum(S)        
        return scores
        
    
    
    def _cal_limb(self, knns, Y, k):
        """
        calulate and Micro Local Imbalance of input dataset, 
        knns the knn of each row
        Y is the interaction matrix
        k is the number of neighbours
        """
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            for j in range(Y.shape[1]):
                if Y[i,j] == 1: # only consider "1" 
                    C[i,j] = k-np.sum(Y[ii,j])
        C = C/k
        milb = np.sum(C)/np.sum(Y)
        
        return milb    

    
    def __str__(self):
        return "Model: KNN: kd={:d}, kt={:d}".format(self.kd,self.kt) 
    

