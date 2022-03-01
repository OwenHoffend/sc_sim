from multiprocessing.sharedctypes import Value
import numpy as np
import copy
from sim.PTM import *
from sim.circuits_obj import *

def get_SEC_class(func, nc, nv, k, consts):
    #Expects the circuit function's inputs to be ordered with all constants first
    assert len(consts) == nc
    Mf = get_func_mat(func, nc + nv, k)
    for idx, const in enumerate(consts):
        Mf = reduce_func_mat(Mf, idx, const)
    return Mf @ B_mat(k) 

def append_to_all(all_poss, n):
    for a in all_poss:
        a.append(n)
    return all_poss

def add_poss(all_poss, new_poss):
    combined_poss = []
    for nwp in new_poss:
        combined_poss += append_to_all(copy.deepcopy(all_poss), nwp)
    return combined_poss

def get_SECs(func, nc, nv, k, consts):
    #Get all of the stochastically equivalent circuits given a SEC description for a circuit with 1 constant input
    #and n variable inputs.
    SEC_class = get_SEC_class(func, nc, nv, k, consts)
    vc = get_vin_mc0(consts)

    poss_width = k * (2 ** nc)
    poss_bvals = B_mat(poss_width)
    #poss = np.zeros((2 ** poss_width, k))
    poss_dict = {}
    for i in range(2 ** poss_width):
        brow = poss_bvals[i, :].reshape(2 ** nc, k)
        row_key = str(tuple(vc.T @ brow))
        if row_key in poss_dict:
            poss_dict[row_key].append(brow)
        else:
            poss_dict[row_key] = [brow,]
    all_poss = [[]]
    for i, a in enumerate(SEC_class):
        new_poss = poss_dict[str(tuple(a))]
        all_poss = add_poss(all_poss, new_poss)
    return all_poss

def SEC_corr_score(a, o1_idx, o2_idx):
    s = 0
    for a_ in a:
        s += np.sum(a_[:, o1_idx] & a_[:, o2_idx])
    return s

def SEC_corr_score_K(K1, K2):
    return np.sum(np.bitwise_and(K1, K2))

def opt_K_max(K):
    _, tlen = K.shape
    K_sum = np.sum(K, axis=1)
    return np.stack([np.pad(np.ones(t, dtype=np.bool_), (0, tlen-t), 'constant') for t in K_sum])

def max_corr_2outputs_restricted(cir, o1_idx=0, o2_idx=1):
    #Directly compute a circuit design that maximizes the output correlation between two inputs
    #Restrict the constant inputs to be the value "0.5" only
    Mf = cir.ptm()
    A = Mf @ B_mat(cir.k) #2**(nc+nv) x k
    K1 = A[:, o1_idx].reshape(2**cir.nc, 2**cir.nv).T
    K2 = A[:, o2_idx].reshape(2**cir.nc, 2**cir.nv).T
    print("# overlaps before: ", SEC_corr_score_K(K1, K2))
    K1_opt = opt_K_max(K1)
    K2_opt = opt_K_max(K2)
    print("# overlaps after: ", SEC_corr_score_K(K1_opt, K2_opt))