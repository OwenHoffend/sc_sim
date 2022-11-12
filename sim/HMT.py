import numpy as np
import matplotlib.pyplot as plt
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

def test_HMT_corr_opt():
    """Generate a bunch of HMTs of various different sizes and evaluate the benefit obtained from SEC optimization"""
    #Run parameters
    precision = 32
    num_weights = 9
    k = 2
    num_tests = 10000
    num_area_iters = 3
    io = IO_Params(ilog2(precision), num_weights, k)
    
    for _ in range(num_area_iters):
        funcs = []
        consts_mat = np.zeros((num_weights, k))
        for i in range(k):
            r = np.random.rand(num_weights)
            consts = np.floor(precision * r / np.sum(r)).astype(int)
            consts_mat[:, i] = consts
            funcs.append(HMT(consts, precision))
        ptm_orig, ptm_opt = opt_max_multi(funcs, io)

        orig_cost = espresso_get_SOP_area(ptm_orig, "hmt.in")
        opt_cost = espresso_get_SOP_area(ptm_opt, "hmt.in")
        print('orig cost: ', orig_cost)
        print('opt cost: ', opt_cost)
        print(consts_mat)
        if opt_cost / orig_cost < 2:
            break

    #Tests for correctness & cout
    correct_func = lambda x: (x.T @ (consts_mat / precision)).T
    test_avg_corr(ptm_orig, ptm_opt, xfunc_uniform(num_weights), correct_func, num_tests, io)

    correct_func_AND = lambda x: np.min(correct_func(x))
    ptm_l2 = get_func_mat(np.bitwise_and, 2, 1)
    test_avg_err(ptm_orig @ ptm_l2, ptm_opt @ ptm_l2, xfunc_uniform(num_weights), correct_func_AND, num_tests, io)

    test_avg_corr(ptm_orig, ptm_opt, xfunc_3x3_img_windows(), correct_func, num_tests, io)
    test_avg_err(ptm_orig @ ptm_l2, ptm_opt @ ptm_l2, xfunc_3x3_img_windows(), correct_func_AND, num_tests, io)

def test_HMT():
    precision = 64
    num_tests = 1000
    depth = ilog2(precision)
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