import numpy as np
from base.inbase import InductiveModelBase
from tree.ebictree import EBiCTree
from mf.nrlmf import NRLMF
from base.clone import clone
from sklearn.preprocessing import MinMaxScaler

class EBiCTreeR(InductiveModelBase):
    """
    Implementataion of Ensembles of Bi-Clustering trees with output space reconstruction
    Drug-target interaction prediction with treeensemble learning and output space reconstruction, BMC Bioinformatics, 2020
    """
    def __init__(self, l_model= EBiCTree(splr=1, splc=1, n_models = 100, seed=0), r_model = NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)):
        self.l_model = l_model # model for learning and predicting, ensemble of Bi-Clustering trees
        self.r_model = r_model # model for output space reconstruction
            
        self.copyable_attrs = ['l_model','r_model']
        
        
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        
        self.r_model.fit(intMat, drugMat, targetMat, cvs)
        re_intMat = self.r_model.predict_trainset()
        re_intMat = MinMaxScaler().fit_transform(re_intMat)
        re_intMat[intMat == 1] = 1
        self.l_model.fit(re_intMat, drugMat, targetMat, cvs)
        
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores = self.l_model.predict(drugMatTe, targetMatTe)
        return scores
    
    