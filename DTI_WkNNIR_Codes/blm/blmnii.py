import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from base.inbase import InductiveModelBase


class BLMNII(InductiveModelBase):
    """
    Implementation of BLMNII: 
    [1] Mei, Jian-Ping, et al. "Drug target interaction prediction by learning from local information and neighbors." Bioinformatics 29.2 (2013): 238-245.
    [2] van Laarhoven, Twan, Sander B. Nabuurs, and Elena Marchiori. "Gaussian interaction profile kernels for predicting drug-target interaction." Bioinformatics 27.21 (2011): 3036-3043.    
    """

    def __init__(self, alpha=0.5, gamma=1.0, sigma=1.0, avg=True):
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.avg = avg
            
        self.copyable_attrs = ['alpha','gamma','sigma','avg']    
      
    def _kernel_combination(self, Y, S, bandwidth):
        # Y: interaction matrix, S: chemical (drug)/genomic (target) similarity matrix
        K = self.alpha*S+(1.0-self.alpha)*rbf_kernel(Y, gamma=bandwidth)
        return K
    
    def _rls_train(self, Y, K): # RLS-avg classifier
        """ return the optimal parameter of the model: W*=(K+σI)^-1Y """
        W = np.linalg.inv(K+self.sigma*np.eye(K.shape[0])) #(K+σI)^-1
        W = np.dot(W, Y)
        return W 

    def _rls_predict(self, W, K):
        return np.dot(K,W) # return the predictions where K is the test kernel, W is the model parameters
    
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        
        self._intMat = intMat
        n1 = np.sum(self._intMat) # the number of "1"s in intMat 
        self._drug_bw = self.gamma*self._n_drugs/n1
        self._target_bw = self.gamma*self._n_targets/n1
        
        
        self._K_d = self._kernel_combination(intMat,drugMat, self._drug_bw)
        self._K_t = self._kernel_combination(intMat.T,targetMat,self._target_bw)
        if self._cvs == 2 or self._cvs == 3:
            self._W_t = self._rls_train(self._intMat, self._K_d) 
            # W_t is the target-based model: feature is K_d, label is target; W_t.shape: n_drugs, n_targets
            self._W_d = self._rls_train(self._intMat.T, self._K_t) 
            # W_d is the drug-based model: feature is K_t, label is drug; W_d.shape: n_targets, n_drugs
        
        
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        if self._cvs == 2:
            # new drug: learn new drug-based models (W_d_te) for predicting new drugs
            # compute the estimated Y: Id 
            Id = np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            for d in range(self._n_drugs_te):
                Id[d,:] = np.dot(drugMatTe[d,:],self._intMat)
                x1, x2 = np.max(Id[d, :]), np.min(Id[d, :])
                Id[d, :] = (Id[d, :]-x2)/(x1-x2)
            K_t_te = self._K_t # target is known, the training and test target kernels are some
            W_d_te = self._rls_train(Id.T,self._K_t)
            # known target: use trained target-based models (W_t) for predicting
            K_d_te = drugMatTe
            W_t_te = self._W_t
        elif self._cvs == 3:
            # known drug: use trained drug-based models (W_d) for predicting
            K_t_te = targetMatTe
            W_d_te = self._W_d
            # new target: learn new target-based models (W_d_te) for predicting new targets
            # compute the estimated Y: It 
            It = np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            for t in range(self._n_targets_te):
                It[:,t] = np.dot(self._intMat,targetMatTe[t,:])
                x1, x2 = np.max(It[:,t]), np.min(It[:,t])
                It[:,t] = (It[:,t]-x2)/(x1-x2)
            K_d_te = self._K_d # drug is known, the training and test drug kernels are some
            W_t_te = self._rls_train(It,self._K_d)
        elif self._cvs == 4:
            # new drug: learn new drug-based models (W_d_te) for predicting new drugs
            # compute the estimated Y: Id 
            Id = np.zeros((self._n_drugs_te,self._n_targets),dtype=float)
            for d in range(self._n_drugs_te):
                Id[d,:] = np.dot(drugMatTe[d,:],self._intMat)
                x1, x2 = np.max(Id[d, :]), np.min(Id[d, :])
                Id[d, :] = (Id[d, :]-x2)/(x1-x2)
            K_t_te = targetMatTe
            W_d_te = self._rls_train(Id.T,self._K_t)
            # new target: learn new target-based models (W_d_te) for predicting new targets
            # compute the estimated Y: It 
            It = np.zeros((self._n_drugs,self._n_targets_te),dtype=float)
            for t in range(self._n_targets_te):
                It[:,t] = np.dot(self._intMat,targetMatTe[t,:])
                x1, x2 = np.max(It[:,t]), np.min(It[:,t])
                It[:,t] = (It[:,t]-x2)/(x1-x2)
            K_d_te = drugMatTe
            W_t_te = self._rls_train(It,self._K_d)
            
        
        scores_d = np.transpose(self._rls_predict(W_d_te, K_t_te)) # drug-based model (feature is target, label is drug)
        scores_t = self._rls_predict(W_t_te, K_d_te) # target-based model (feature is drug, label is target)
        
        if self.avg:
            scores = 0.5*(scores_d+scores_t)
        else:
            scores = np.maximum(scores_d, scores_t)
        return scores
      
    
    def __str__(self):
        return "Model:BLMNII, alpha:%s, gamma:%s, sigma:%s, avg:%s" % (self.alpha, self.gamma, self.sigma, self.avg)
