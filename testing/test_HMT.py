import logging
logging.basicConfig(encoding='utf-8', level=logging.DEBUG, handlers=[
    logging.FileHandler('HMT_results/hmt_logs.log'),
    logging.StreamHandler()
])
import numpy as np
from sim.HMT import *
from sim.PTM import *
from sim.SEC_opt_macros import *
from multiprocessing import Pool

def test_HMT_corr_opt_sweep():
    funcs = [False, True] #two different funcs, but one needs to switch between 2x2 and 3x3 windows
    ps = [32, ]
    ks = [2, 3, 4]
    nvs = [4, 9]
    results = np.empty((len(funcs)*len(ps)*len(ks)*len(nvs), 6))
    idx = 0
    for func in funcs:
        for p in ps:
            for k in ks:
                for nv in nvs:
                    if func:
                        xfunc = xfunc_3x3_img_windows() if nv == 9 else xfunc_2x2_img_windows()
                    else:
                        xfunc = xfunc_uniform(nv)
                    s = "func: {}, p: {}, k: {}, nv: {}".format(func, p, k, nv)
                    logging.info(s)
                    results[idx, :] = test_HMT_corr_opt(nv, k, p, xfunc)
    np.save("HMT_results/hmt_area_opt.npy", results)

def test_HMT_corr_opt(num_weights, k, p, xfunc):
    """Generate a bunch of HMTs of various different sizes and evaluate the benefit obtained from SEC optimization"""
    #Run parameters
    proc = 0
    precision = p
    num_tests = 1000
    num_area_iters = 100
    #good_threshold = 3
    io = IO_Params(ilog2(precision), num_weights, k)
    
    ptm_orig, ptm_opt = None, None

    orig_cost = 0
    opt_cost = 0
    opt_cost_area = 0 
    orig_corr = 0 
    opt_corr = 0
    opt_corr_area = 0
    for ai in range(num_area_iters):
        funcs, consts_mat = get_random_HMT(num_weights, precision, k)
        #print(consts_mat)

        ptm_orig, ptm_opt = opt_multi(funcs, io, opt_K_multi)
        _, ptm_opt_area = opt_multi(funcs, io, opt_K_max_area_aware_multi)

        #--- AREA COST ---
        orig_cost += espresso_get_SOP_area(ptm_orig, "hmt{}.in".format(proc))
        opt_cost += espresso_get_SOP_area(ptm_opt, "hmt{}.in".format(proc))
        opt_cost_area += espresso_get_SOP_area(ptm_opt_area, "hmt{}.in".format(proc))

        #--- CORRECTNESS & OUTPUT CORRELATION --- 
        correct_func = lambda x: (x.T @ (consts_mat / precision)).T
        #test_avg_corr(ptm_orig, ptm_opt, xfunc_uniform(num_weights), num_tests, io, correct_funct=correct_func)

        orig_corr_, opt_corr_ = test_avg_corr(ptm_orig, ptm_opt, xfunc, num_tests, io, correct_func=correct_func)
        _, opt_corr_area_ = test_avg_corr(ptm_orig, ptm_opt_area, xfunc, num_tests, io, correct_func=correct_func)

        orig_corr += 2 * np.sum(np.tril(orig_corr_, -1)) / (k ** 2 - k)
        opt_corr += 2 * np.sum(np.tril(opt_corr_, -1)) / (k ** 2 - k)
        opt_corr_area += 2 * np.sum(np.tril(opt_corr_area_, -1)) / (k ** 2 - k)

        #print("Cout improvement: ", np.abs(cout_avg_ - cout_opt_avg_))

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
        
    orig_cost /= num_area_iters
    opt_cost /= num_area_iters
    opt_cost_area /= num_area_iters
    logging.info('orig cost: {}'.format(orig_cost))
    logging.info('opt cost: {}'.format(opt_cost))
    logging.info('opt cost area: {}'.format(opt_cost_area))

    orig_corr /= num_area_iters
    opt_corr /= num_area_iters
    opt_corr_area /= num_area_iters
    logging.info('orig corr: {}'.format(orig_corr))
    logging.info('opt corr: {}'.format(opt_corr))
    logging.info('opt corr area: {}'.format(opt_corr_area))
    return np.array([orig_cost, opt_cost, opt_cost_area, orig_corr, opt_corr, opt_corr_area])

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