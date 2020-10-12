import numpy as np
from base.inbase import InductiveModelBase
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
from scipy.spatial import distance


class ECKNN():
    """
    Implementation of BLM_ECKNN which is the base method for ALADIN
    """
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, Y):
        """
        X: the features matrix of training data set
        Y: the label matrix of training data set
        """
        # get correct labels Yc
        self._Yc = np.zeros(Y.shape, dtype=float)
        self._neigh = NearestNeighbors(n_neighbors=self.k)
        self._neigh.fit(X)
        knns = self._neigh.kneighbors(return_distance=False) # the knns of X (the fitted examples)
        # get reverse neighbours
        rns = []
        for i in range(Y.shape[0]):
            rns.append([])
        for i in range(Y.shape[0]):
            for j in knns[i]:
                rns[j].append(i)
        # get corrected labels        
        for i in range(Y.shape[0]):
            rn = rns[i]
            n_rn = len(rns[i])
            if n_rn > 0:
                self._Yc[i,:] = np.sum(Y[rn,:],axis=0)
                self._Yc[i,:] /= n_rn
            else:
                self._Yc[i,:] = Y[i,:]
                
        
    def predict(self, X):
        """
        X: the features matrix of test data set
        """
        scores = np.zeros((X.shape[0], self._Yc.shape[1]), dtype=float)
        knns = self._neigh.kneighbors(X, return_distance=False)
        for i in range(X.shape[0]): #  each test instance
            ii = knns[i]
            scores[i,:] = np.sum(self._Yc[ii,:], axis=0)
        scores /= self.k
        return scores
    


class ALADIN(InductiveModelBase):
    """
    Implementation of ALADIN: 
    Buza K, Peska L. Aladin: A new approach for drug–target interaction prediction[C]. Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Cham, 2017: 322-337
    Buza, K., Nanopoulos, A., Nagy, G.: Nearest neighbor regression in the presence of bad hubs. Knowl.-Based Syst. 86, 250–260 (2015)
       
    ALADIN could deal with S2 and S3 but fail to S4
    
    Do not use enhanced feature is better than use it, because the elements of test interaction matrix are all "0"s,
    which makes the jaccard similarity between training and test drugs (targets) unreliable
    """

    def __init__(self, k=3, n_sf = 20, n_models = 25, is_ef=False, seed=0):
        self.k = k
        self.n_sf = n_sf  # number of selected fetures, if n_sf == -1, select all features
        self.n_models = n_models # number of models
        self.is_ef = is_ef # If use enhanced features or not
        self.seed = seed
            
        self.copyable_attrs = ['k','n_sf','n_models','is_ef','seed']
     
        
    def _local_model_train(self, X, Y):
        # X is features, Y is labels
        models = [] 
        findexs = []  # the index of feature subset
        for i in range(self.n_models):
            # get sub features
            if self.n_sf == -1 or self.n_sf >= X.shape[1]: # select all features
                X_sub = X  # the features of target-based model
            else:    
                prng = np.random.RandomState(self.seed+i)
                index_t = prng.choice(X.shape[1], self.n_sf, replace=False)
                X_sub= X[:,index_t]   # the features of target-based model
                findexs.append(index_t)
            mt = ECKNN(k= self.k) # train target-based model
            mt.fit(X_sub, Y)
            models.append(mt)  
        return models, findexs
    
    
    def _local_model_predict(self, models, findexs, X_te, scores):
        # X is test features
        for i in range(self.n_models):
            if self.n_sf == -1 or self.n_sf >= X_te.shape[1]:
                Xt_sub = X_te
            else:
                index_t = findexs[i]
                Xt_sub= X_te[:,index_t]
            scores += models[i].predict(Xt_sub)
        scores /= self.n_models
        return scores
    
    
    def fit(self, intMat, drugMat, targetMat, cvs=2): 
        self._check_fit_data(intMat, drugMat, targetMat, cvs)        
        
        if self._cvs == 2 and self.k > self._n_drugs:
            self.k = self._n_drugs
        if self._cvs == 3 and self.k > self._n_targets:
            self.k = self._n_targets
        if self.is_ef:
            if self.n_sf > 2*min(self._n_drugs,self._n_targets):
                self.n_sf = -1 # select all features
        else:
            if self.n_sf > min(self._n_drugs,self._n_targets):
                self.n_sf = -1 # select all features
        
        self._drugMat = drugMat
        self._targetMat = targetMat
        self._intMat = intMat
        
        if self._cvs == 2 or self._cvs == 4:
            # train target-based model
            if self.is_ef:
                SId = distance.cdist(intMat, intMat, 'jaccard') # drug similarity from interaction matrix     
                Xt = np.hstack((drugMat, SId)) # enchanced drug features for target-based model
            else:
                Xt = drugMat
            Yt = intMat  # the labels of target-based model 
            self._model_ts, self._findex_ts = self._local_model_train(Xt, Yt)
            # self._model_ts : target-based models, target is label
            # self._findex_ts :  # the index of feature subset of target-based models
            
        
        if self._cvs == 3 or self._cvs == 4:
            # train drug-based model
            if self.is_ef:
                SIt = distance.cdist(intMat.T, intMat.T, 'jaccard') # target similarity from interaction matrix   
                Xd = np.hstack((targetMat, SIt)) # enchanced target features for drug-based model 
            else:
                Xd = targetMat
            Yd = intMat.T # the labels of drug-based model 
            self._model_ds, self._findex_ds = self._local_model_train(Xd, Yd)
            # self._model_ds:  drug-based models, drug is label 
            # self._findex_ds:  the index of feature subset of drug-based models
    
                     
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        # scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
         
        if self._cvs == 2:  # new drug, known target
            scores_d=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            scores_t=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            # new drug, use WP instead of self._model_ds 
            for d in range(self._n_drugs_te): # d: each test drug
                for d_tr in range(self._n_drugs): # d_tr: each training drug
                    scores_d[d,:] += drugMatTe[d,d_tr]*self._intMat[d_tr,:]
                scores_d[d,:]/=np.sum(drugMatTe[d,])
            # known target, use self._model_ts
            if self.is_ef:
                intMat_te = np.zeros(scores_t.shape, dtype=float)
                SId = distance.cdist(intMat_te, self._intMat, 'jaccard') # drug similarity from interaction matrix
                Xt = np.hstack((drugMatTe, SId)) # enchanced drug features for target-based model
            else:
                Xt = drugMatTe
            scores_t = self._local_model_predict(self._model_ts, self._findex_ts, Xt, scores_t)    
            
            
        elif self._cvs == 3: # known drug, new target
            scores_d=np.zeros((self._n_targets_te,self._n_drugs_te),dtype=float)
            scores_t=np.zeros((self._n_targets_te,self._n_drugs_te),dtype=float)
            # known drug, use self._model_ds
            if self.is_ef:
                intMat_te = np.zeros(scores_d.shape, dtype=float)
                SIt = distance.cdist(intMat_te, self._intMat.T, 'jaccard') # drug similarity from interaction matrix
                Xd = np.hstack((targetMatTe, SIt)) # enchanced drug features for target-based model
            else:
                Xd = targetMatTe
            scores_d = self._local_model_predict(self._model_ds, self._findex_ds ,Xd, scores_d)  
            
            # new target, use WP instead of self._model_ts
            for t in range(self._n_targets_te): # d: each test target
                for t_tr in range(self._n_targets): # d_tr: each training target
                    scores_t[t,:] += targetMatTe[t,t_tr]*self._intMat[:,t_tr]
                scores_t[t,:]/=np.sum(targetMatTe[t,])
            scores_d = scores_d.T
            scores_t = scores_t.T
            
            
        elif self._cvs == 4:
            scores_d=np.zeros((self._n_targets_te,self._n_drugs_te),dtype=float)
            scores_t=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            
            Y2 = np.zeros((self._n_drugs_te, self._n_targets),dtype=float)
            if self.is_ef:
                intMat_te = np.zeros(Y2.shape, dtype=float)
                SId = distance.cdist(intMat_te, self._intMat, 'jaccard') # drug similarity from interaction matrix
                Xt = np.hstack((drugMatTe, SId)) # enchanced drug features for target-based model
            else:
                Xt = drugMatTe
            Y2 = self._local_model_predict(self._model_ts, self._findex_ts, Xt, Y2)
            # train drug-based models where targetMat is training features, Y2.T is training labels, targetMatTe is test features
            # do not use enhanced features because the Y2 is not binary matrix that could not compute jaccard similarity
            model_ds2, findex_ds2 = self._local_model_train(self._targetMat, Y2.T)
            scores_d = self._local_model_predict(model_ds2, findex_ds2, targetMatTe, scores_d)   
            scores_d = scores_d.T
            
            Y3 = np.zeros((self._n_targets_te, self._n_drugs), dtype=float)
            if self.is_ef:
                intMat_te = np.zeros(Y3.shape, dtype=float)
                SIt = distance.cdist(intMat_te, self._intMat.T, 'jaccard') # drug similarity from interaction matrix
                Xd = np.hstack((targetMatTe, SIt)) # enchanced drug features for target-based model
            else:
                Xd = targetMatTe
            Y3 = self._local_model_predict(self._model_ds, self._findex_ds ,Xd, Y3)  
            # train target-based models where drugMat is training features, Y3.T is training labels, drugMatTe is test features
            # do not use enhanced features because the Y3 is not binary matrix that could not compute jaccard similarity
            model_ts2, findex_ts2 = self._local_model_train(self._drugMat, Y3.T)
            scores_t = self._local_model_predict(model_ts2, findex_ts2, drugMatTe, scores_t)  

        scores = (scores_d+scores_t)/2
        return scores      
              