import numpy as np
from sim.PTM import *

"""Code to generate a design that represents a "canonical" correlation-optimal circuit for the case where SCC=1"""
def therm_encoding(*Cs):
    c_int = int_array(np.array(Cs))
    return np.pad(np.ones(c_int, dtype=np.bool_), (2**len(Cs)-c_int, 0), 'constant') #2 ** nc paddings

def weight_gen(*Vs, K=None):
    assert K is not None
    v_int = int_array(np.array(Vs))
    return np.sum(K[v_int, :])

def canonical_max_scc(*Ins, K=None, nv=None):
    assert nv is not None
    Vs = Ins[:nv]
    Cs = Ins[nv:]
    weight = weight_gen(*Vs, K=K)
    therms = therm_encoding(*Cs)
    return therms[weight]

# def weight_gen(*Vs, K=None):
#    assert K is not None
#    nv2, _ = K.shape
#    weights = np.sum(K, axis=1)
#    unq = np.unique(weights)
#    weight_As = np.empty((nv2, len(unq)), dtype=np.bool_)
#    print("Weight gen compression ratio: ", nv2 / len(unq)) #compression ratio
#    for i in range(nv2):
#        j = weights[i]
#        weight_As[i, j] = True #project weight
#
#    v_int = int_array(np.array(Vs)[::-1])
#    return tuple(weight_As[v_int,:])