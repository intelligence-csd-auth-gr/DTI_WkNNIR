from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
import numpy as np

class MLKNN:
    """ Implementation of MLkNN: 
        Min-Ling Zhang, Zhi-Hua Zhou. ML-KNN: A lazy learning approach to multi-label learning[J].
        Pattern Recognition, 40(7):2038-2048.     
    """
    def __init__(self, k=5, s=1.0):
        self.k = k
        self.s = s
    
    def fit(self, X, Y):
        """ X is the similarities matrix between training instances, where diagnol elements are "0" (minimun value) """
        self._X_dis = 1-X  # self._X_dis is the distance matrix 
        self._Y = Y
        self._num_ins = X.shape[0]
        self._num_labels = Y.shape[1]
        self._neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        self._neigh.fit(np.zeros((self._num_ins,self._num_ins)))
        self._p_prob1, self._p_prob0 = self.compute_prior()
        self._c_prob1, self._c_prob0 = self.compute_cond()
    
    def compute_prior(self):
        p_prob1 = (self.s+np.sum(self._Y,axis=0))/(2*self.s+self._num_ins)
        p_prob0 = 1-p_prob1
        return p_prob1, p_prob0
    
    def compute_cond(self):
        c1= np.zeros((self._num_labels,self.k+1),dtype=int)
        c0= np.zeros((self._num_labels,self.k+1),dtype=int)
        knn_indices = self._neigh.kneighbors(self._X_dis, return_distance=False)
        for i in range(self._num_ins):
            ii = knn_indices[i]
            deltas = np.sum(self._Y[ii,:], axis=0, dtype=int) # number of "1"s in kNNs for every labels 
            for j in range(self._num_labels):
                if self._Y[i,j] == 1:
                    c1[j, deltas[j]]+=1
                else:
                    c0[j, deltas[j]]+=1
                    
        sum_c1= np.sum(c1, axis=1)            
        sum_c0= np.sum(c0, axis=1)
        c_prob1=np.zeros((self._num_labels, self.k + 1), dtype=float)
        c_prob0=np.zeros((self._num_labels, self.k + 1), dtype=float)
        for i in range(self._num_labels):
            for j in range(self.k+1):
                c_prob1[i,j] =(self.s+c1[i,j])/(self.s*(self.k+1)+sum_c1[i])
                c_prob0[i,j] =(self.s+c0[i,j])/(self.s*(self.k+1)+sum_c0[i])
        
        return c_prob1, c_prob0
    
    def predict(self, X):
        """ X is the simialrity matrix between trainning instance and test instance 
            X.#column == self.X.#row 
        """
        X_dis = 1 - X # X is the distance matrix between trainning instance and test instance 
        num_ins_te = X.shape[0]
        scores = np.zeros((num_ins_te, self._num_labels),dtype=float)
        labels = np.zeros((num_ins_te, self._num_labels),dtype=int)
        knn_indices = self._neigh.kneighbors(X_dis, return_distance=False)
        for i in range(num_ins_te):
            ii = knn_indices[i]
            deltas = np.sum(self._Y[ii,:], axis=0, dtype=int) # number of "1"s in kNNs for every labels 
            for j in range(self._num_labels):
                p1 = self._p_prob1[j] * self._c_prob1[j, deltas[j]]
                p0 = self._p_prob0[j] * self._c_prob0[j, deltas[j]]
                scores[i,j]=p1/(p1+p0)
                labels[i,j] = int(p1 >= p0)
        # print(np.sum(labels))        
        return scores, labels
        