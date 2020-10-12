import numpy as np
from base.inbase import InductiveModelBase
from tree.bictree import BiCTree
from base.clone import clone

class EBiCTree(InductiveModelBase):
    """
    Implementataion of Ensembles of Bi-Clustering tree model
    Pliakos, K. and Vens, C., Network inference with ensembles of bi-clustering trees, BMC Bioinformatics, 2019
    """
    def __init__(self, splr=1, splc=1, n_models = 10, seed=0):
        self.n_models = n_models
        self.splr = splr # The minimum number of row samples (drugs) required to split an internal node.
        self.splc = splc # The minimum number of column samples (target) required to split an internal node.
        self.seed = seed    
        self.copyable_attrs = ['n_models','splr','splc','seed'] 
        
    
    def _initilize_model(self):
        if self.n_models < 1:
            self.n_models = 1
        self._models = []
        tree = BiCTree(splr=self.splr, splc=self.splc, isRandomTree = True)
        for i in range(self.n_models):
            m = clone(tree)
            m.seed = self.seed + i
            self._models.append(m)

    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self._initilize_model()
        for i in range(self.n_models):
            self._models[i].fit(intMat, drugMat, targetMat, self._cvs)
            
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        for i in range(self.n_models):
            scores += self._models[i].predict(drugMatTe,targetMatTe)
        scores /= self.n_models
        return scores