import numpy as np


from scipy.cluster.hierarchy import linkage,fcluster 
from scipy.spatial.distance import squareform 

from base.inbase import InductiveModelBase
from mll.mlknn import MLKNN
from collections import defaultdict


class MLKNNSC(InductiveModelBase):
    """
    Implementation of MLkNNSC: 
    Shi J Y , Yiu S M , Li Y , et al. Predicting drug-target interaction for new drugs using enhanced similarity measures and super-target clustering1[J]. Methods, 2015, 83(C):98-104.
    """
    def __init__(self, k=5, s=1.0, c=1.1):
        self.k = k
        self.s = s
        self.c = c # cut-off of agglomerative hierarchical clustering
        self.copyable_attrs = ['k','s','c']

    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        if (self._cvs == 2 or self._cvs == 4) and self._n_drugs < self.k:
            self.k = self._n_drugs
        elif (self._cvs == 3 or self._cvs == 4) and self._n_targets < self.k:
            self.k = self._n_targets
        
        self._intMat = intMat
        self._drugMat = (drugMat + drugMat.T)/2 # ensure self._drug_ids is symmetric matrix
        self._drugMat = self._drugMat - np.diag(np.diag(self._drugMat)) # set all diagnol elements to 0
        self._targetMat  = (targetMat + targetMat.T)/2
        self._targetMat = self._targetMat - np.diag(np.diag(self._targetMat))

        if self._cvs == 2 or self._cvs == 4:
            # target-based model (drug is feature, target is label)
            self._model_t = MLKNN(self.k, self.s)  
            self._clu_dict_target, self._clu_target, sY_t = self._get_super_label(self._targetMat,self._intMat)
            Yt = np.hstack((self._intMat,sY_t))   # Yd.shape = num_drugs, num_targets+num_super_targets 
            self._model_t.fit(self._drugMat, Yt)
            
        if self._cvs == 3 or self._cvs == 4:
            # drug-based model (target is feature, drug is label)
            self._model_d = MLKNN(self.k, self.s)  
            self._clu_dict_drug, self._clu_drug, sY_d = self._get_super_label(self._drugMat,self._intMat.T)
            Yd = np.hstack((self._intMat.T,sY_d))  # Yd.shape = num_targets, num_drugs+num_super_drugs
            self._model_d.fit(self._targetMat, Yd)

    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        if self._cvs == 2: # new drug--knonw target, using target-based model
            scores_t,_=self._model_t.predict(drugMatTe)
            scores = scores_t[:,:self._n_targets]
            for i in range(self._n_targets): # each known target
                clu_id = self._clu_target[i] # the id of cluster which i-th target belongs to
                scores[:,i] *= scores_t[:,self._n_targets + clu_id]      
        elif self._cvs == 3: # known drug--new target, using drug-based model
            scores_d,_=self._model_d.predict(targetMatTe)
            scores = scores_d[:,:self._n_drugs]
            for i in range(self._n_drugs): # each known drug
                clu_id = self._clu_drug[i] # the id of cluster which i-th drug belongs to
                scores[:,i] *= scores_d[:,self._n_drugs + clu_id]
            scores = scores.T
        elif self._cvs == 4: # new drug--new target
            scores_t,_= self._model_t.predict(drugMatTe)
            clu_st_id = self._cluster_predict(targetMatTe, self._clu_dict_target)
            ii = [x+self._n_targets for x in clu_st_id]
            scores_t2 = scores_t[:, ii]   
            scores_d,_= self._model_d.predict(targetMatTe)
            clu_sd_id = self._cluster_predict(drugMatTe, self._clu_dict_drug)
            jj = [x+self._n_drugs for x in clu_sd_id]
            scores_d2 = scores_d[:, jj]
            scores = (scores_t2 + scores_d2.T)/2
        
        return scores
            
    def _cluster_predict(self, simMat, clu_dict):
        """ return the list, where i-th element is the cluster id (start from 0) of i-th test instance"""
        # simMat: the similarity matrix between new instances and training instances
        # i.e. simMat[i,j] is the similarity between i-th new instance and j-th training instance
        num = simMat.shape[0]
        clu = []
        for i in range(num):
            s_max = -1
            id_max = -1
            for clu_id, ii in clu_dict.items():
                s = np.mean(simMat[i,ii]) 
                if s > s_max:
                    s_max = s
                    id_max =clu_id
            clu.append(id_max)
        return clu
        
        
    def _get_super_label(self, S, Y):
        # S is the similarity matrix of labels Y
        disMat = 1 - S
        disMat = disMat-np.diag(np.diag(disMat)) # set the diagnoal element of disMat to 0
        clu = self._hierarchy_cluster(disMat)
        clu_dict = defaultdict(list) # key: cluster-id (start from 0), value: indices of instances belonging to key cluster
        for i in range(len(clu)):
            clu_dict[clu[i]].append(i)
        num_sY=len(clu_dict)
        
        sY= np.zeros((Y.shape[0],num_sY),dtype=float)
        for i in range(num_sY):
            ii = clu_dict[i]  # the indices of instances belonging to i-th cluster
            sY[:,i] = np.sum(Y[:,ii],axis=1)
        sY[sY > 1] = 1 # set elements that are larger than 1 to 1
        
        return clu_dict, clu, sY

    def _hierarchy_cluster(self, disMat):
        """ return the ndarray (shape 1,n), where i-th element is the cluster id (start from 0) of i-th instances """
        disVec = squareform(disMat) # the input matrix's diagnoal must be 0 of squareform
        Z=linkage(disVec,'ward')
        clu=fcluster(Z, t=self.c, criterion='distance')  # cluster id starts from 1
        clu = clu -1  # cluster id starts from 0, i.e. clu[0]=1 the 0-th instances belongs to i-th cluster
        return clu

        
    def __str__(self):
        return "Model: MLKNNSC, k:{:d}, s:{:f}, c:{:f}".format(self.k, self.s, self.c)