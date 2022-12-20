import numpy as np
import matplotlib.pyplot as plt
from sim.PTM import *
from sim.SEC_opt_macros import *
from sim.circuits import mux_1
from multiprocessing import Pool

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

def test_HMT_corr_opt(proc):
    """Generate a bunch of HMTs of various different sizes and evaluate the benefit obtained from SEC optimization"""
    #Run parameters
    precision = 8
    num_weights = 4
    k = 4
    num_tests = 10000
    num_area_iters = 50
    xfunc = xfunc_uniform
    #good_threshold = 3
    io = IO_Params(ilog2(precision), num_weights, k)
    
    ptm_orig, ptm_opt = None, None
    #good_ones = []
    #good_ptms = []
    #ifactors = []
    for ai in range(num_area_iters):
        print("proc: {}, iter: {}".format(proc, ai))

        funcs, consts_mat = get_random_HMT(num_weights, precision, k)
        print(consts_mat)

        ptm_orig, ptm_opt = opt_multi(funcs, io, opt_K_multi)
        _, ptm_opt_area = opt_multi(funcs, io, opt_K_max_area_aware_multi)

        #--- AREA COST ---
        orig_cost = espresso_get_SOP_area(ptm_orig, "hmt{}.in".format(proc))
        opt_cost = espresso_get_SOP_area(ptm_opt, "hmt{}.in".format(proc))
        opt_cost_area = espresso_get_SOP_area(ptm_opt_area, "hmt{}.in".format(proc))
        print('orig cost: ', orig_cost)
        print('opt cost: ', opt_cost)
        print('opt cost area: ', opt_cost_area)
        #Select the ensemble min
        costs = [opt_cost, opt_cost_area]
        m_ind = np.argmin(costs)
        opt_cost = costs[m_ind]
        ptm_opt = [ptm_opt, ptm_opt_area][m_ind]
        cost_ratio = opt_cost / orig_cost #Want this to be low
        print("Cost ratio: ", cost_ratio)

        #--- CORRECTNESS & OUTPUT CORRELATION --- 
        correct_func = lambda x: (x.T @ (consts_mat / precision)).T
        #test_avg_corr(ptm_orig, ptm_opt, xfunc_uniform(num_weights), num_tests, io, correct_funct=correct_func)

        cout_avg, cout_opt_avg = test_avg_corr(ptm_orig, ptm_opt, xfunc(num_weights), num_tests, io, correct_func=correct_func)


        #--- OUTPUT ERROR TEST - RELU --- 
        #   ptm_l2 = get_func_mat(lambda x, y: np.bitwise_and(x, np.bitwise_not(y)), 2, 1) #ReLU type subtraction between two layers
        #   correct_func_RELU = lambda x: np.maximum(correct_func(x)[0] - correct_func(x)[1], 0)
        #   c_err, c_err_opt = test_avg_err(ptm_orig @ ptm_l2, ptm_opt @ ptm_l2, xfunc(num_weights), correct_func_RELU, num_tests, io)
        #   c_err_ratio = c_err / c_err_opt  #Want this to be high
        #   improvement_factor = c_err_ratio / cost_ratio
        #   print('orig err: ', c_err)
        #   print('opt err: ', c_err_opt)
        #   #print("Correlation error ratio:", c_err_ratio)
        #   print("Improvement factor: ", improvement_factor)

        #   if improvement_factor >= good_threshold:
        #       good_ones.append(consts_mat)
        #       good_ptms.append(ptm_opt)
        #       ifactors.append(improvement_factor)
    #np.save("./HMT_results/results{}.npy".format(proc), np.array(good_ones))
    #np.save("./HMT_results/ptms{}.npy".format(proc), np.array(good_ptms))
    #np.save("./HMT_results/ifactors{}.npy".format(proc), np.array(ifactors))

def test_HMT_corr_opt_mp(f):
        with Pool(16) as p:
            p.map(f, list(range(16)))

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