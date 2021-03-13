from base.inbase import InductiveModelBase
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
import numpy as np


class WKNKNRI4_(InductiveModelBase):
    """
    This is the updated WKNKNRI method (https://arxiv.org/pdf/2012.12325v2.pdf)
    Difference with WKNKNIR: 
        1. recovery interaction by using NCd as weighting matrix
        2. Ydt are used for all settings (S2, S3, S4)
        3. add exponent for similarities with lower local imbalance of corresponding intMat submatrix
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
    
        self._neigh_d = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        self._neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
        self._neigh_t = NearestNeighbors(n_neighbors=self.kr, metric='precomputed')
        self._neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
        self._knns_t = self._neigh_t.kneighbors(1 - self._targetMat, return_distance=False)   
        self._knns_d = self._neigh_d.kneighbors(1 - self._drugMat, return_distance=False)
        
        limb_d, self._Cd, NCd = self._cal_limb(self._knns_d, self._intMat, self.k)
        limb_t, self._Ct, NCt = self._cal_limb(self._knns_t, self._intMat.T, self.k)
        self._Ct = self._Ct.T        
        NCd, NCt = np.exp(-NCd), np.exp(-NCt)
        
        Yt = self._recover_intMat(self._knns_t, self._targetMat, self._intMat.T)*(NCt)
        Yt = Yt.T
        Yd = self._recover_intMat(self._knns_d, self._drugMat, self._intMat)*(NCd)
        Ydt = (Yt+Yd)/2
        self._intMat = np.maximum(Ydt, self._intMat)         
    #--------------------------------------------------------------------------------------
        
    def _recover_intMat(self, knns, S, Y):
        Yr = np.zeros(Y.shape)
        etas = self.T**np.arange(self.k)
        for d in range(Y.shape[0]): 
            ii = knns[d]
            sd = S[d,ii]
            Yr[d,:] = etas*sd@Y[ii,:]
            z = np.sum(sd)
            if z>0:
                Yr[d,:]/=z                
        # Yd[Yd>1] = 1
        return Yr
    #--------------------------------------------------------------------------------------
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)

        if self._cvs == 2:  # new drug, known target
            ii_d = self._neigh_d.kneighbors(1 - drugMatTe, return_distance=False)
            etas = self.T**np.arange(self.k)
            for d in range(self._n_drugs_te): # d: each test drug
                ii = ii_d[d]
                sd = drugMatTe[d,ii]
                scores[d,:] = etas*sd@self._intMat[ii,:]
                z = np.sum(sd)
                if z>0:
                    scores[d,:]/=z
        elif self._cvs == 3: # known drug, new target
            jj_t = self._neigh_t.kneighbors(1 - targetMatTe, return_distance=False)
            etas = self.T**np.arange(self.k)
            for t in range(self._n_targets_te):
                jj = jj_t[t]
                st = targetMatTe[t,jj]
                scores[:,t] = etas*st@(self._intMat[:,jj]).T
                z = np.sum(st)
                if z>0:
                    scores[:,t]/=z
        elif self._cvs == 4:
            scores = self._predict_cvs4_1_22(drugMatTe, targetMatTe)
                                 
        return scores
    #--------------------------------------------------------------------------------------  

    def _predict_cvs4_1_22(self, drugMatTe, targetMatTe):
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
       
        ii_d = self._neigh_d.kneighbors(1 - drugMatTe, return_distance=False)
        jj_t = self._neigh_t.kneighbors(1 - targetMatTe, return_distance=False)
        
        W = np.ones((self.k,self.k))
        for i in range(self.k):
            W[i,:] *= self.T**(i) #self.T**(i/ri)
        for j in range(self.k):
            W[:,j] *= self.T**(j)  # self.T**(j/rj)
        
        for d in range(self._n_drugs_te):
            ii = ii_d[d]
            for t in range(self._n_targets_te):
                jj = jj_t[t]
                idx = np.ix_(ii,jj) # submatrix indices
                
                # get submatirx of Cd and Ct for ii and jj
                LId = np.sum(self._Cd[idx])
                LIt = np.sum(self._Ct[idx])                
                if LId == 0 or LIt == 0:
                    S = np.outer(drugMatTe[d,ii],targetMatTe[t,jj])
                    Y = W*S*self._intMat[idx]
                    scores[d,t] = np.sum(Y)
                    z = np.sum(S)
                    if z>0:
                        scores[d,t]/=z
                else:        
                    r = LId/LIt
                    sd = drugMatTe[d,ii]
                    st = targetMatTe[t,jj]
                    if r<=1:
                        ri, rj = r, 1.0
                        sd = sd**ri
                    else:
                        ri, rj = 1.0, 1/r
                        st = st**rj
                    S = np.outer(sd,st)
                    Y = W*S*self._intMat[idx]
                    scores[d,t] = np.sum(Y)
                    z = np.sum(S)
                    if z>0:
                        scores[d,t]/=z           
        return scores
    #--------------------------------------------------------------------------------------  
        
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
            C[i,:] = k-np.sum(Y[ii,:], axis=0)
        C[Y==0] = 0 # only consider "1" 
        C = C/k
        milb = np.sum(C[Y==1])/np.sum(Y)
        
        NC = np.zeros(C.shape)
        for i in range(Y.shape[0]):
            ii = knns[i]
            NC[i] = np.sum(C[ii,:], axis=0)
            y = np.sum(Y[ii,:], axis=0)
            y[y==0] = 1
            NC[i] /= y
        
        return milb, C, NC
    #--------------------------------------------------------------------------------------      

    
    def __str__(self):
        return "Model: KNN: kd={:d}, kt={:d}".format(self.kd,self.kt) 
#--------------------------------------------------------------------------------------    
  