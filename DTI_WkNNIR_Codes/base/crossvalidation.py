# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:34:58 2020

@author: lb
"""
import numpy as np
import time
import copy
import itertools

from collections import defaultdict 
from base.clone import clone
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc, average_precision_score
from joblib import Parallel, delayed


def cross_validate_innerCV_params(model, cvs, num, intMat, drugMat, targetMat, seeds=[0], out_file_name=None, n_jobs=5, **params_cd):
    inner_num = 5 # the number of fold for inner CV
    if intMat.shape[0]+intMat.shape[1]<100: # small size dataset as nr
        inner_num = 10
    if cvs == 4:
        inner_num = 2
        if intMat.shape[0]+intMat.shape[1]<100: # small size dataset as nr
            inner_num = 3
    aupr, auc, times, params_list = [], [], [], []
    cv_data = cv_split_index(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    for seed, cv_index in cv_data.items():       
        for train_d,train_t,test_d,test_t in cv_index:
            train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMat, targetMat,cvs)
            # _,_,_, params = cross_validate_various_params(model, cvs, inner_num, train_Y, train_dMat, train_tMat, seeds=[0], out_file_name=None, **params_cd)
            _,_,_, params = cross_validate_various_params_parallel(model, cvs, inner_num, train_Y, train_dMat, train_tMat, seeds=[0], out_file_name=None, n_jobs=n_jobs, **params_cd)
            # params is the best parameter setting choosen by the inner CV for current fold
            m = clone(model)
            m.set_params(**params)
            tic=time.time()
            m.fit(train_Y, train_dMat, train_tMat, cvs)
            scores = m.predict(test_dMat, test_tMat)
            run_time = time.time()-tic
            aupr_val, auc_val = cal_metrics(scores, test_Y)
            aupr.append(aupr_val)
            auc.append(auc_val)
            times.append(run_time)
            params_list.append(params)
            if out_file_name is not None:
                with open(out_file_name,"a") as f:
                    f.write("{:.6f}\t{:.6f}\t{:.6f}\t{}\n".format(aupr_val, auc_val, run_time, params)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    
    return auprs, aucs, times, params_list
    

    
def cross_validate_various_params_parallel(model, cvs, num, intMat, drugMat, targetMat, seeds=[0], out_file_name=None, n_jobs=5, **params_cd):
    best_aupr, best_auc, best_run_time = 0, 0, 0
    best_param = dict()
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    params_names = []
    params = []
    for k,v in params_cd.items():    
        params_names.append(k)
        params.append(v)
    param_list =[]
    for params_values in itertools.product(*params):
        param = dict()
        for i in range(len(params_names)):
            param[params_names[i]] = params_values[i]     
        # the param is the one parammeter setting of params_cd, i.e. param={'kd':5,'kt':7}
        param_list.append(param)
     
    num_params = len(param_list)
    if num_params < n_jobs:
        n_jobs = num_params
    result_list = Parallel(n_jobs=n_jobs)(delayed(run_cv_with_params)(model, cvs, num, intMat, drugMat, targetMat, seeds, None, param) for param in param_list)
    for i in range(len(result_list)):
        aupr, auc, run_time = result_list[i]
        param = param_list[i]
        if out_file_name is not None:                
            with open(out_file_name,"a") as f:
                f.write('{:.6f}\t{:.6f}\t{:6f}\t{}\n'.format(aupr, auc, run_time, param)) 
        if aupr > best_aupr:
            best_aupr = aupr
            best_auc = auc
            best_run_time = run_time
            best_param = param            
    if out_file_name is not None:                
        with open(out_file_name,"a") as f:
            f.write('\n\nOptimal parammeter setting:\n')
            f.write('{:.6f}\t{:.6f}\t{:6f}\t{}\n'.format(best_aupr,best_auc, best_run_time, best_param))      
    return best_aupr, best_auc, best_run_time, best_param  
        
                
def run_cv_with_params(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name, param):
    model.set_params(**param)
    auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name)
    return auprs.mean(), aucs.mean(), run_times.mean()


def cross_validate(model, cvs, num, intMat, drugMat, targetMat, seeds=[0], out_file_name=None, n_jobs=5):
    """    
    parammeters
    ----------
    model : InductiveModel. 
    cvs : int, cvs = 2,3 or 4
        The setting of cross validation.
    num : int 
        the number of the fold for CV.
    intMat : ndarray
        interaction matrix.
    drugMat : ndarray
        drug similairty matrix.
    targetMat : ndarray
        target  similairty matrix.
    seed: int
        the seed used to split the dataset. The default is None.
    out_file_name : string, optional
        the out put file out detail results of CV. If it is None, do not out put detial results. The default is None.
        
    Returns
    -------
    auprs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUPR results for each fold.
    aucs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    times : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.

    """
    aupr, auc, times = [], [], []
    cv_data = cv_split_index(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    for seed, cv_index in cv_data.items():       
        for train_d,train_t,test_d,test_t in cv_index:
            train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMat, targetMat,cvs)
            m = clone(model)
            tic=time.time()
            m.fit(train_Y, train_dMat, train_tMat, cvs)
            scores = m.predict(test_dMat, test_tMat)
            run_time = time.time()-tic
            aupr_val, auc_val = cal_metrics(scores, test_Y)            
            aupr.append(aupr_val)
            auc.append(auc_val)
            times.append(run_time)
            if out_file_name is not None:
                with open(out_file_name,"a") as f:
                    f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(aupr_val, auc_val, run_time)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    
    return auprs, aucs, times

def train_test_model(model, cvs, intMat, drugMat, targetMat, train_d, train_t, test_d, test_t):
    train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMat, targetMat,cvs)
    m = clone(model)
    tic=time.time()
    m.fit(train_Y, train_dMat, train_tMat, cvs)
    scores = m.predict(test_dMat, test_tMat)
    run_time = time.time()-tic
    aupr_val, auc_val = cal_metrics(scores, test_Y)
    return aupr_val, auc_val, run_time 


def cross_validate_parallel(model, cvs, num, intMat, drugMat, targetMat, seeds=[0], out_file_name=None, n_jobs=5):
    """    
    parammeters
    ----------
    model : InductiveModel. 
    cvs : int, cvs = 2,3 or 4
        The setting of cross validation.
    num : int 
        the number of the fold for CV.
    intMat : ndarray
        interaction matrix.
    drugMat : ndarray
        drug similairty matrix.
    targetMat : ndarray
        target  similairty matrix.
    seed: int
        the seed used to split the dataset. The default is None.
    out_file_name : string, optional
        the out put file out detail results of CV. If it is None, do not out put detial results. The default is None.
        
    Returns
    -------
    auprs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUPR results for each fold.
    aucs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    times : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    """
    aupr, auc, times = [], [], []
    cv_data = cv_split_index(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    cv_index = []
    for seed, cv_index1 in cv_data.items():
        cv_index.extend(cv_index1)
        
    result_list = Parallel(n_jobs=n_jobs)(delayed(train_test_model)(model, cvs, intMat, drugMat, targetMat, train_d,train_t,test_d,test_t) for (train_d,train_t,test_d,test_t) in cv_index)
    
    for aupr_val, auc_val, run_time in result_list:
        aupr.append(aupr_val)
        auc.append(auc_val)
        times.append(run_time)
        if out_file_name is not None:
            with open(out_file_name,"a") as f:
                f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(aupr_val, auc_val, run_time)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    return auprs, aucs, times


def train_test_model_en(model, cvs, intMat, drugMat, targetMat, train_d, train_t, test_d, test_t, param=None):
    train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMat, targetMat,cvs)
    m = clone(model)    
    if param is not None:
        m.base_model.set_params(**param)
    
    tic=time.time()
    m.fit(train_Y, train_dMat, train_tMat, cvs)
    scores = m.predict(test_dMat, test_tMat)
    run_time = time.time()-tic
    aupr_val, auc_val = cal_metrics(scores, test_Y)
    return aupr_val, auc_val, run_time 



def cross_validate_parallel_en(model, cvs, num, intMat, drugMat, targetMat, seeds=[0], out_file_name=None, n_jobs=5, params = None):
    """    
    parammeters
    ----------
    model : InductiveModel. 
    cvs : int, cvs = 2,3 or 4
        The setting of cross validation.
    num : int 
        the number of the fold for CV.
    intMat : ndarray
        interaction matrix.
    drugMat : ndarray
        drug similairty matrix.
    targetMat : ndarray
        target  similairty matrix.
    seed: int
        the seed used to split the dataset. The default is None.
    out_file_name : string, optional
        the out put file out detail results of CV. If it is None, do not out put detial results. The default is None.
    params: dict list
        the params of base learner for each CV fold
    
    Returns
    -------
    auprs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUPR results for each fold.
    aucs : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    times : ndarray, shape (len(seeds)*num), len(seeds)*num is the total number of CV experiments conducted
        the AUC results for each fold.
    """
    aupr, auc, times = [], [], []
    cv_data = cv_split_index(intMat, cvs, num, seeds)
    if out_file_name is not None:
        with open(out_file_name,"w") as f:
            f.write("AUPR\tAUC\tTime\tParams\n")
    cv_index = []
    for seed, cv_index1 in cv_data.items():
        cv_index.extend(cv_index1)
    
    cv_index_param = []
    for i in range(len(cv_index)):
        cv_index_param.append((cv_index[i], params[i]))

    
    result_list = Parallel(n_jobs=n_jobs)(delayed(train_test_model_en)(model, cvs, intMat, drugMat, targetMat, train_d,train_t,test_d,test_t, param) for ((train_d,train_t,test_d,test_t),param) in cv_index_param)
    
    for aupr_val, auc_val, run_time in result_list:
        aupr.append(aupr_val)
        auc.append(auc_val)
        times.append(run_time)
        if out_file_name is not None:
            with open(out_file_name,"a") as f:
                f.write("{:.6f}\t{:.6f}\t{:.6f}\n".format(aupr_val, auc_val, run_time)) 
    auprs = np.array(aupr, dtype=np.float64)
    aucs = np.array(auc, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    return auprs, aucs, times





def cv_split_index(intMat, cvs, num, seeds): # num: the number of folds
    """ 
    Inductive Learning, CV for Setting 4 according to 'Toward more realistic drug-target interaction predictions' 2014
    num=3: rowFold=[0,1,2] columnFold=[0,1,2], 9 test set: [r0-c0, r0-c1, r0-c2,...,r2-c2]
    """
    if cvs==1:
        print ("Inductive Learning cannot handle Setting 1 !!!")
        return None
    
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cvs == 2:  # Setting 2  
            index_d = prng.permutation(num_drugs)
            cv_index_d = np.array_split(index_d,num)   
            for i in range(num):
                train_d=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_d=np.concatenate((train_d, cv_index_d[j]))
                    else:
                        test_d=cv_index_d[j]
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                cv_data[seed].append((train_d,None,test_d,None))    
                # (drug indices in training set, taraget indices in training set, drug indices in training set, target indices in test set) 
                # None: all drugs (rows) / targets (columns) are used in training/test set
        elif cvs == 3: # Setting 3 
            index_t = prng.permutation(num_targets)
            cv_index_t = np.array_split(index_t,num)
            for i in range(num):
                train_t=np.array([], dtype=np.intc)
                for j in range(num):
                    if j!= i:
                        train_t=np.concatenate((train_t, cv_index_t[j]))
                    else:
                        test_t=cv_index_t[j]
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((None,train_t,None,test_t))               
        elif cvs == 4: # Setting 4
            index_d = prng.permutation(num_drugs)
            index_t = prng.permutation(num_targets)
            cv_index_d = np.array_split(index_d,num)
            cv_index_t = np.array_split(index_t,num)
            folds=[(i_r,i_c) for i_r in range(num) for i_c in range(num)]
            for i_r, i_c in folds:
                train_d = np.array([], dtype=np.intc)
                train_t = np.array([], dtype=np.intc)
                test_d = cv_index_d[i_r]
                test_t = cv_index_t[i_c]
                for j_r in range(num):
                    if(i_r != j_r):
                        train_d=np.concatenate((train_d, cv_index_d[j_r]))
                for j_c in range(num):
                    if(i_c != j_c):
                        train_t=np.concatenate((train_t, cv_index_t[j_c]))
                train_d=np.sort(train_d)
                test_d=np.sort(test_d)
                train_t=np.sort(train_t)
                test_t=np.sort(test_t)
                cv_data[seed].append((train_d,train_t,test_d,test_t))    

    # for seed, cv_index in cv_data.items():
    #     print(seed)
    #     for train_d,train_t,test_d,test_t in cv_index:
    #         print(test_d)
            
    return cv_data


def split_train_test_set(train_d,train_t,test_d,test_t, yMat, dMat, tMat, cvs): 
    """ split the original data into training and test data according to the splitting indices."""
    if cvs==1:
        print ("Inductive Learning cannot handle Setting 1 !!!")
        return None
    
    if cvs==2:
        train_dMat = dMat[np.ix_(train_d,train_d)]
        test_dMat = dMat[np.ix_(test_d,train_d)]
        train_tMat = tMat
        test_tMat  = tMat
        train_Y = yMat[train_d,:]
        test_Y  = yMat[test_d,:]  
    elif cvs==3:
        train_dMat=dMat
        test_dMat=dMat
        train_tMat = tMat[np.ix_(train_t,train_t)]
        test_tMat  = tMat[np.ix_(test_t,train_t)] 
        train_Y = yMat[:,train_t]
        test_Y  = yMat[:,test_t] 
    elif cvs==4:
        train_dMat = dMat[np.ix_(train_d,train_d)]
        test_dMat = dMat[np.ix_(test_d,train_d)]
        train_tMat = tMat[np.ix_(train_t,train_t)]
        test_tMat  = tMat[np.ix_(test_t,train_t)] 
        train_Y = yMat[np.ix_(train_d,train_t)]
        test_Y  = yMat[np.ix_(test_d,test_t)]
        
    return train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y


def cal_aupr(scores_1d, labels_1d):
    """ scores_1d and labels_1d is 1 dimensional array """
    # prec, rec, thr = precision_recall_curve(labels_1d, scores_1d)
    # aupr_val =  auc(rec, prec)
    # This kind of calculation is not right
    aupr_val = average_precision_score(labels_1d,scores_1d)
    if np.isnan(aupr_val):
        aupr_val=0.0
    return aupr_val

        
def cal_auc(scores_1d, labels_1d):    
    """ scores_1d and labels_1d is 1 dimensional array """
    fpr, tpr, thr = roc_curve(labels_1d, scores_1d)
    auc_val = auc(fpr, tpr)
    # same with:    roc_auc_score(labels,scores)
    if np.isnan(auc_val):
        auc_val=0.5
    return auc_val


def cal_metrics(scores, labels):
    """ scores and labels are 2 dimensional matrix, return aucpr, auc"""
    scores_1d=scores.flatten()
    labels_1d=labels.flatten()
    aupr_val=cal_aupr(scores_1d, labels_1d)
    auc_val= cal_auc(scores_1d, labels_1d)
    return aupr_val, auc_val