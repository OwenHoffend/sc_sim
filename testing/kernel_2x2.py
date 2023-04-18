#Fresh implementation of the 2x2 convolution kernel testing - the old ones are really cluttered
from json import load
import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1
from sim.SEC_opt_macros import *
import matplotlib.pyplot as plt

def pcc(cs, val, precision):
    if val == 0:
        return False
    elif val == 1:
        return True
    elif val == 0.5:
        return cs[0]

    radix_bits = np.zeros(precision, dtype=np.bool_)
    cmp = 0.5
    for i in range(precision):
        if val >= cmp:
            radix_bits[i] = 1
            val -= cmp
        else:
            radix_bits[i] = 0
        cmp /= 2
    while radix_bits[-1] == 0:
        radix_bits = radix_bits[:-1]
    precision = radix_bits.size
    actual_precision = precision
    assert len(cs) >= actual_precision
    result = cs[0]

    radix_bits = radix_bits[:-1][::-1]
    for i in range(actual_precision-1):
        bit = radix_bits[i]
        if bit:
            result = np.bitwise_or(result, cs[i+1])
        else:
            result = np.bitwise_and(result, cs[i+1])
    return result

def kernel_2x2(*x, precision = 3,
    kernel = [
        0.125, 0.875,
        0.125, 0.125
    ]
):
    xv = x[0:4]
    xc = x[4:4+precision]
    s = x[-1]
    m = np.empty(4, dtype=np.bool_)
    for i in range(4):
        m[i] = np.bitwise_and(xv[i], pcc(xc, kernel[i], precision))

    top = mux_1(s, m[0], m[1])
    bot = mux_1(s, m[2], m[3])
    return top, bot

def test_kernel_2x2():
    precision = 8
    num_kernels = 25
    num_tests = 100
    kernels = np.zeros((num_kernels, 4))
    corrs = np.zeros((num_kernels, 2))
    errs = np.zeros((num_kernels, 2))
    areas = np.zeros((num_kernels, 2))

    #Simulation options
    use_ptm = False
    N = 2**(precision+1)

    for kernel_idx in range(num_kernels): 
        kernel = np.random.randint(0, 2**precision, 4).astype(np.float32) / 2**precision
        kernels[kernel_idx, :] = kernel
        ptm = get_func_mat(lambda *x: kernel_2x2(*x, precision=precision, kernel=kernel), precision+5, 2)
        io = IO_Params(precision+1, 4, 2)
        Ks = get_Ks_from_ptm(ptm, io)
        K1, K2 = opt_K_max(Ks[0]), opt_K_max(Ks[1])
        ptm_opt = Ks_to_Mf([K1, K2])
        def xfunc():
            xs = np.random.rand(4)
            #xs[2], xs[3] = xs[0], xs[1] #this line makes the circuit the EXACT SAME as the one from my DDECS paper
            return xs
        
        def l1_correct_func(px):
            return 0.5 * np.array([px[0] * kernel[0] + px[1] * kernel[1], px[2] * kernel[2] + px[3] * kernel[3]])
        def l2_correct_func(px):
            l1 = l1_correct_func(px)
            return np.maximum(0, l1[0] - l1[1])
        
        print("Kernel idx: ", kernel_idx)
        print("Kernel: ", kernel)

        #PTM based evaluation
        corr, corr_opt = test_avg_corr(ptm, ptm_opt, xfunc, num_tests, io, correct_func=l1_correct_func, use_ptm=use_ptm, N=N)
        corrs[kernel_idx, 0], corrs[kernel_idx, 1] = corr[1,0], corr_opt[1,0]

        relu = SeriesCircuit([ParallelCircuit([NOT(), I(1)]), AND()]).ptm()
        err, err_opt = test_avg_err(ptm @ relu, ptm_opt @ relu, xfunc, l2_correct_func, num_tests, io, use_ptm=use_ptm, N=N)
        errs[kernel_idx, 0], errs[kernel_idx, 1] = err, err_opt

        areas[kernel_idx, 0] = espresso_get_SOP_area(ptm, "mux_opt.in")
        areas[kernel_idx, 1] = espresso_get_SOP_area(ptm_opt, "mux_opt.in")

    print(np.mean(errs, axis=0))

    np.save("2x2_kernels.npy", kernels) #D_ means the DDECs style design, the ones without it have 4 variable inputs not just 2
    np.save("2x2_corrs.npy", corrs)
    np.save("2x2_errs.npy", errs)
    np.save("2x2_areas.npy", areas)