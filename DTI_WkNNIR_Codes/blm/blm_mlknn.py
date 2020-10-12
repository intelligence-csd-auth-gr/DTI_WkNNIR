from base.inbase import InductiveModelBase
from mll.mlknn import MLKNN
import numpy as np

class BLM_MLKNN(InductiveModelBase):
    """ BLM using MLkNN as the base learner """    
    def __init__(self, k=3, s=1.0):
        self.k = k
        self.s = s
        self.copyable_attrs = ['k','s']

    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        if self._cvs == 2 and self._n_drugs < self.k:
            self.k = self._n_drugs
        elif self._cvs == 3 and self._n_targets < self.k:
            self.k = self._n_targets
        elif self._cvs == 4 and min(self._n_targets, self._n_drugs) < self.k:
            self.k = min(self._n_targets, self._n_drugs)

        if self._cvs == 2 or self._cvs == 4:
            self._drugMat = drugMat - np.diag(np.diag(drugMat)) # drug similarity matrix where diagnol elements are 0
            self._model_t = MLKNN(k=self.k, s=self.s) # target-based model, drug is feature, target is label
            self._model_t.fit(self._drugMat, intMat)
        if self._cvs == 3 or self._cvs == 4:
            self._targetMat = targetMat - np.diag(np.diag(targetMat))
            self._model_d = MLKNN(k=self.k, s=self.s) # drug-based model, target is feature, drug is label
            self._model_d.fit(self._targetMat, intMat.T)
        
        
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        if self._cvs == 2: # new drug--knonw target, using target-based model
            scores_t,_=self._model_t.predict(drugMatTe)
            scores = scores_t
        elif self._cvs == 3: # known drug--new target, using drug-based model
            scores_d,_=self._model_d.predict(targetMatTe)
            scores = scores_d.T
        elif self._cvs == 4: #new drug--new target
            # get estimated interaction matrices for new drugs and new targets
            _,Y2 =self._model_t.predict(drugMatTe) # Y2: #drug_test * #target_train
            _,Y3 =self._model_d.predict(targetMatTe) 
            Y3 = Y3.T # Y3: #drug_train * #target_test 
            # trainning models that using new drugs (targets) as labels
            model_2= MLKNN(self.k, self.s)  # targetMat is feature, Y2.T (new drug) is label
            model_2.fit(self._targetMat,Y2.T)
            scores_2,_ = model_2.predict(targetMatTe)
            model_3= MLKNN(self.k, self.s)  # drugMat is feature, Y3 (new target) is label
            model_3.fit(self._drugMat,Y3)
            scores_3,_ = model_3.predict(drugMatTe)
            scores = (scores_2.T+ scores_3)/2
            
        return scores
    
    def __str__(self):
        return "Model: BLM_MLKNN, k:{:d}, s:{:f}".format(self.k, self.s)


