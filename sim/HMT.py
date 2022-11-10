import numpy as np
import matplotlib.pyplot as plt
from sim.PTM import *
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
    max_height = np.log2(precision)
    assert np.ceil(max_height) == max_height #Precision must be a power of 2
    max_height = max_height.astype(int)
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
                    print("MUX : s{}".format(layer))
                else: #Add an AND gate
                    oq.append(np.bitwise_and(layer_s, a))
                    print("AND : s{}".format(layer))
            print("Layer {} has {} outputs".format(layer, len(oq)))
        assert len(oq) == 1
        return oq[0]
    return f

def test_HMT_corr_opt():
    """Generate a bunch of HMTs of various different sizes and evaluate the benefit obtained from SEC optimization"""
    pass

def test_HMT():
    precision = 64
    num_tests = 1000
    depth = np.log2(precision).astype(int)
    consts = np.array([10, 13, 1, 1, 1, 2, 7, 9])
    n = consts.size + depth
    f = HMT(consts, precision)
    ptm = get_func_mat(f, n, 1)
    for _ in range(num_tests):
        xvals = np.random.uniform(size=consts.size)
        inputs = np.concatenate([xvals, np.array([0.5 for _ in range(depth)])])
        vin = get_vin_mc0(inputs)
        correct = xvals.T @ (consts / precision)
        test = ptm.T @ vin
        print(correct)
        print(test)
        assert np.isclose(test[1], correct, 1e-6)