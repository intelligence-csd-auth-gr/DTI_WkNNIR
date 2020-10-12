import os
import time
import sys
import numpy as np

from base.crossvalidation import *
from base.loaddata import * 

from blm.aladin import ALADIN
from blm.blm_mlknn import BLM_MLKNN
from blm.blmnii import BLMNII
from nn.mlknnsc import MLKNNSC
from mf.nrlmf import NRLMF
from nn.wknkn import WKNKN
from nn.wknknri import WKNKNRI
from tree.bictree import BiCTree
from tree.ebictree import EBiCTree
from tree.ebictreer import EBiCTreeR
from en.ers import ERS
from en.egs import EGS
from en.els import ELS


"""
'wknkn' or WKNKN corresponds to the WKNN method in the paper
'wknknri' or WKNKNRI corresponds to the WkNNIR method in the paper 
'ers' or ERS corresponds to the ERS ensemble method in the paper
'egs' or EGS corresponds to the EGS ensemble method in the paper
'els' or ELS corresponds to the ELS ensemble method in the paper
"""

def initialize_model(method, cvs=2):
    model = None
    if method == 'blm_mlknn':
        model = BLM_MLKNN(k=3, s=1.0)
    elif method == 'mlknnsc':
        model = MLKNNSC(k=3, s=1.0, c=1.1)
    elif method == 'blmnii':
        model = BLMNII(alpha=0.5, gamma=1.0, sigma=1.0, avg=True)
    elif method == 'nrlmf':
        model = NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
    elif method == 'aladin':
        model = ALADIN(k=3, n_sf = 20, n_models = 25, is_ef=False)
    elif method == 'ebictree':
        model = EBiCTree(splr=1, splc=1, n_models = 100, seed = 0)
    elif method == 'ebictreer': 
        nrlmf = NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
        if cvs == 2:
            ebictree = EBiCTree(splr=2, splc=2, n_models = 100, seed = 0)
        else:
            ebictree = EBiCTree(splr=1, splc=1, n_models = 100, seed = 0)
        model = EBiCTreeR(l_model = ebictree, r_model= nrlmf)
    elif method == 'wknkn':
        model = WKNKN(k=7, T=0.8)
    elif method == 'wknknri':
        model = WKNKNRI(k=7, kr=7, T=0.8)
    elif method == 'ers':
        model = ERS(base_model = initialize_model('wknknri'), n_models=30, seed=0, p=0.95)
    elif method == 'egs':
        model = EGS(base_model = initialize_model('wknknri'), n_models=30, seed=0, p=0.95)
    elif method == 'els':
        model = ELS(base_model = initialize_model('wknknri'), n_models=30, seed=0, p=0.95, k=5)   
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model function!!".format(method))
    return model

def initialize_parameter_candidates(method, cvs=2):
    params_cd = dict()
    if method == 'blm_mlknn':
        params_cd = {'k':[1,2,3,5,7,9]}
    elif method == 'mlknnsc':
        params_cd = {'k':[1,2,3,5,7,9]}
    elif method == 'blm_mlknnls':
        params_cd = {'kd':[1,2,3,5,7,9],'kt':[1,2,3,5,7,9]}
    elif method == 'blmnii':
        values_a = np.arange(0,1.1,0.1) # array([0.1,0.2...,0.9,1.0])
        values_s = [2**-5,2**-4,2**-3,2**-2,2**-1,1.0]
        params_cd = {'alpha':values_a, 'gamma':[1.0], 'sigma':values_s, 'avg':[True]}
    elif method == 'nrlmf':
        values = np.arange(-5, 1)
        values = 2.0**values
        params_cd = {'num_factors':[50,100],'lambda_d': values, 'alpha':values, 'beta':values}
    elif method == 'aladin':
        params_cd = {'k':[1,2,3,5,7,9] ,'n_sf':[10,20,50]}
    elif method == 'wknkn':
        values = np.arange(0.1,1.1,0.1)
        params_cd = {'k':[1,2,3,5,7,9],'T':values} #'k':[1,2,3,5,7,9] 'k':[10,15,20]
    elif method == 'wknknri':
        values = np.arange(0.1,1.1,0.1)
        params_cd = {'k':[1,2,3,5,7,9] ,'kr':[1],'T':values} #'k':[1,2,3,5,7,9]  [5,10,15,20,25,30]
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_parameter_candidates function!!".format(method))
    return params_cd

if __name__ == "__main__": 
    # !!! my_path should be change to the path of the project in your machine            
    my_path = 'F:\envs\DTI_WkNNIR\DTI_WkNNIR'   
    n_jobs = 5
    
    sys.path.append(os.path.join(my_path, 'DTI_WkNNIR_Codes'))  
    
    data_dir =  os.path.join(my_path, 'datasets') #'F:\envs\DPI_Py37\Codes_Py3\datasets' #'data'
    output_dir = os.path.join(my_path, 'output') #'F:\envs\DPI_Py37\Codes_Py3\output' 
    seeds = [0,1]
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 
    out_summary_file = os.path.join(output_dir, "summary_result"+ time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()) +".txt")

    parameter_setting = 0
    # 0: run cross validation on a model with specific parameters setting 
    # 2: run cross validation on a model with choosing parammeters by inner CV on training set for each fold
    # 5 run cross validation on ensemble models with various base learners, and the parameters of base learners are retirved from files
    if parameter_setting == 0:
        """ run cross validation on a model with specific parameters setting"""
        cmd ="Method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime"
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        for method in ['ebictreer']: 
            for cvs in [2,3,4]: # 2,3,4
                model = initialize_model(method,cvs)  # parammeters could be changed in "initialize_model" function    
                num = 10
                if cvs == 4:
                    num = 3
                for dataset in ['nr']:  # 'nr','ic','gpcr','e'
                    out_file_name= os.path.join(output_dir, "Specific_parameters_"+method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                    intMat, drugMat, targetMat = load_data_from_file(dataset, data_dir)
                    tic = time.time()
                    auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name)
                    # auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name, n_jobs)
                    cmd = "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                    print(cmd)
                    with open(out_summary_file,"a") as f:
                        f.write(cmd+"\n")
    elif parameter_setting == 2:
        """ run cross validation on a model with choosing parammeters by inner CV on training set for each fold """
        cmd ="Method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime"
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        for method in ['aladin', 'blmnii', 'mlknnsc', 'blm_mlknn', 'nrlmf', 'wknkn', 'wknknri']:
            model = initialize_model(method)  # parameters could be changed in "initialize_model" function
            for cvs in [2,3,4]: # 2,3,4
                params_cd = initialize_parameter_candidates(method,cvs)
                num = 10
                if cvs == 4:
                    num = 3
                for dataset in ['nr']:  # 'nr','ic','gpcr','e'
                    out_file_name= os.path.join(output_dir, "InnerCV_parammeters_"+method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                    intMat, drugMat, targetMat = load_data_from_file(dataset, data_dir)
                    tic = time.time()
                    auprs, aucs, run_times,_ = cross_validate_innerCV_params(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name, n_jobs, **params_cd)
                    cmd = "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(), time.time()-tic)
                    print(cmd)
                    with open(out_summary_file,"a") as f:
                        f.write(cmd+"\n")
    elif parameter_setting == 5:
        """ run cross validation on ensemble models with various base learners, and the parameters of base learners are retirved from files"""
        cmd ="Method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime"
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        en_methods = ['ers']  # 'ers', 'egs', 'els'
        base_methods = ['aladin', 'blmnii', 'mlknnsc', 'nrlmf', 'wknkn', 'wknknri' ]  #
        model_list = []
        for bm in base_methods:
            b_model = initialize_model(bm)
            for em in en_methods:
                if em is None:
                    model_list.append((bm, b_model))
                else:    
                    e_models = initialize_model(em)
                    e_models.base_model = b_model
                    model_list.append((em, bm, e_models))
        for em, bm, model in model_list:
            method = em+'_'+bm
            for cvs in [2,3,4]: # 2,3,4
                num = 10
                if cvs == 4:
                    num = 3
                for dataset in ['nr']:  # 'nr','ic','gpcr','e'
                    param_file = os.path.join(data_dir,'method_params','InnerCV_parammeters_'+bm+'_S'+str(cvs)+'_'+dataset+'.txt')
                    params = get_params(param_file)
                    
                    out_file_name= os.path.join(output_dir, "Specific_parameters_"+method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                    intMat, drugMat, targetMat = load_data_from_file(dataset, data_dir)
                    tic = time.time()
                    auprs, aucs, run_times = cross_validate_parallel_en(model, cvs, num, intMat, drugMat, targetMat, seeds, out_file_name, n_jobs, params)
                    cmd = "{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                    print(cmd)
                    with open(out_summary_file,"a") as f:
                        f.write(cmd+"\n")                  
    