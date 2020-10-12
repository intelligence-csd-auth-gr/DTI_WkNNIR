import os
import numpy as np
import json

def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        next(inf) # skip the first line
        int_array = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        next(inf) # skip the first line
        drug_sim = [line.strip("\n").split()[1:] for line in inf]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        next(inf) # skip the first line
        target_sim = [line.strip("\n").split()[1:] for line in inf]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix where the row is drug and the column is target
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        firstRow = next(inf)
        drugs = firstRow.strip("\n").split()
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def get_params(param_file):
    params = []
    with open(param_file, "r", encoding='utf-8') as inf:
        next(inf) # skip the 1st line
        for line in inf:
            ss = line.strip().split('\t')
            ss[3] = ss[3].replace(", 'avg': True","" ) # remove the avg parameter for BLMNII
            s = ss[3].replace("'", "\"")
            param = json.loads(s)
            params.append(param)
    return params
            