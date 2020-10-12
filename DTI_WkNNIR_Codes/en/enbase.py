from base.inbase import InductiveModelBase
from base.clone import clone
from nn.wknkn import WKNKN

class EnsembleBase(InductiveModelBase):
    """
    Base class providing API and common functions for all ensemble model.    
    """
    
    def __init__(self, base_model = WKNKN(k=7,T=0.8), n_models=5, seed=0):
        self.base_model = base_model
        self.n_models = n_models
        self.seed = seed
        
        self.copyable_attrs = ['base_model','n_models','seed']
    
    def _initilize_model(self):
        if self.n_models < 1:
            self.n_models = 1
        self._models = []
        self._seeds = []
        for i in range(self.n_models):
            m = clone(self.base_model)
            self._models.append(m)
            self._seeds.append(i+self.seed)
    
                
   
