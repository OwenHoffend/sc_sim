import numpy as np
from sim.PTM import *
import sim.circuits as cir

def get_SEC_class(func, nc, nv, k, consts):
    #Expects the circuit function's inputs to be ordered with all constants first
    assert len(consts) == nc
    Mf = get_func_mat(func, nc + nv, k)
    for idx, const in enumerate(consts):
        Mf = reduce_func_mat(Mf, idx, const)
    return Mf @ B_mat(k) 

def get_SECs(func, nc, nv, k, consts):
    #Get all of the stochastically equivalent circuits given a SEC description for a circuit with 1 constant input
    #and n variable inputs.
    SEC_class = get_SEC_class(func, nc, nv, k, consts)
    vc = get_vin_mc0(consts)
    Bc = B_mat(k * (2 ** nc))
    poss = np.zeros((2 ** (k * (2 ** nc)), k))
    for i in range(k):
        poss[:, i] = vc.T @ Bc[:, i::k].T
    print(poss)
    print(SEC_class)
    a_s = [[] for i in range(2 ** nv)]
    sz = 1
    for i, a in enumerate(SEC_class):
        sub_sz = 0
        for j, po in enumerate(poss):
            if np.all(np.isclose(a, po)):
                sub_sz += 1
                a_s[i].append(Bc[j, :])
        sz *= sub_sz

def SEC_corr_score(A1, A2):
    pass