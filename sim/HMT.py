import numpy as np
from sim.PTM import *
from sim.SEC_opt_macros import *
from sim.circuits import mux_1

#Hardwired MUX tree generation for correlation analysis
def HMT(consts, precision):
    """
        Return a lambda function that implements an HMT with the specified parameters
        consts: An array of integers which represent the numerators to the (positive) weights. 
        precision: The denominator of the weights
    """
    assert np.sum(consts) <= precision #Weights should not sum to more than 1
    num_consts = consts.size
    max_height = ilog2(precision)
    bin_reps = np.zeros((num_consts, max_height), dtype=np.bool_)
    for i, const in enumerate(consts):
        bin_reps[i, :] = bin_array(const, max_height)

    def f(*inputs):
        x = inputs[0:num_consts]
        s = inputs[num_consts:]
        xq = []
        oq = []
        for layer in range(max_height):
            layer_s = s[layer]

            #Add primary inputs to the "x" queue
            for i, b in enumerate(bin_reps[:, layer]):
                if b:
                    xq.append(x[i])
            
            #Take from xq and oq two at a time
            q = xq + oq
            oq = []
            xq = []
            while len(q) > 0:
                a = q.pop(0)
                if len(q) > 0: #Add a MUX
                    b = q.pop(0)
                    oq.append(mux_1(layer_s, a, b))
                    #print("MUX : s{}".format(layer))
                else: #Add an AND gate
                    oq.append(np.bitwise_and(layer_s, a))
                    #print("AND : s{}".format(layer))
            #print("Layer {} has {} outputs".format(layer, len(oq)))
        assert len(oq) == 1
        return oq[0]
    return f

def get_random_HMT(num_weights, precision, k):
    funcs = []
    consts_mat = np.zeros((num_weights, k))
    for i in range(k):
        r = np.random.rand(num_weights)
        consts = np.floor(precision * r / np.sum(r)).astype(int)
        consts_mat[:, i] = consts
        funcs.append(HMT(consts, precision))
    return funcs, consts_mat