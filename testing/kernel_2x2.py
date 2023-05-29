#Fresh implementation of the 2x2 convolution kernel testing - the old ones are really cluttered
from json import load
import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1
from sim.SEC_opt_macros import *
from sim.seq_recorr import *
from sim.verilog_gen import espresso_out_to_verilog
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
    num_kernels = 10
    num_tests = 100
    max_synopsys_tests = 10
    kernels = np.zeros((num_kernels, 4))
    corrs = np.zeros((num_kernels, 2))
    errs = np.zeros((num_kernels, 4))
    areas = np.zeros((num_kernels, 2))

    #Simulation options
    use_ptm = False
    N = 256

    #num_synopsys_tests = 0
    #mean_err = 2.8077
    #mean_area = 1.957
    #std_err = 1.09
    #std_area = 0.
    d = 2
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
        #reco 
        for test_idx in range(num_tests):
            xvals = xfunc()
            correct = l2_correct_func(xvals)
            bs_out, bs_out_opt = compute_pout_sim(ptm, ptm_opt, xvals, io, N)
            bs_out_reco_1, bs_out_reco_2 = fsm_reco_d(bs_out[0, :], bs_out[1, :], d)
            bs_out_opt_reco_1, bs_out_opt_reco_2 = fsm_reco_d(bs_out_opt[0, :], bs_out_opt[1, :], d)
            relu_reco = np.bitwise_and(bs_out_reco_1, np.bitwise_not(bs_out_reco_2))
            relu_reco_opt = np.bitwise_and(bs_out_opt_reco_1, np.bitwise_not(bs_out_opt_reco_2))
            pz_reco = bs.bs_mean(relu_reco, bs_len=N)
            pz_reco_opt = bs.bs_mean(relu_reco_opt, bs_len=N)
            errs[kernel_idx, 2] += (pz_reco - correct) ** 2
            errs[kernel_idx, 3] += (pz_reco_opt - correct) ** 2
        errs[kernel_idx, 2] = np.sqrt(errs[kernel_idx, 2] / num_tests)
        errs[kernel_idx, 3] = np.sqrt(errs[kernel_idx, 3] / num_tests)

        ptm_relu = ptm @ relu
        ptm_relu_opt =  ptm_opt @ relu
        err, err_opt = test_avg_err(ptm_relu, ptm_relu_opt, xfunc, l2_correct_func, num_tests, io, use_ptm=use_ptm, N=N, print_=False)
        errs[kernel_idx, 0], errs[kernel_idx, 1] = err, err_opt

        for i in range(4):
            print(errs[kernel_idx, i])

        #Area analysis
        #fn_orig = f"MAC_relu_{kernel_idx}"
        #fn_opt = f"MAC_relu_opt_{kernel_idx}"
        #areas[kernel_idx, 0] = espresso_get_SOP_area(ptm_relu, fn_orig + ".in")
        #areas[kernel_idx, 1] = espresso_get_SOP_area(ptm_relu_opt, fn_opt + ".in")

        #area_scaling = areas[kernel_idx, 1] / areas[kernel_idx, 0]
        #err_scaling = err / err_opt
        #if mean_err - std_err <= err_scaling <= mean_err + std_err and \
        # mean_area - std_area <= area_scaling <= mean_area + std_area and \
        #    num_synopsys_tests < max_synopsys_tests:
        #    num_synopsys_tests += 1
        #    print("Synopsys test found. Num:", num_synopsys_tests)
        #    espresso_out_to_verilog(fn_orig + ".in", fn_orig)
        #    espresso_out_to_verilog(fn_opt + ".in", fn_opt)

    errs_scaling = errs[:, 0] / errs[:, 1]
    area_scaling = areas[:, 1] / areas[:, 0]
    plt.scatter(errs_scaling, area_scaling)
    plt.show()
    
    m_err_scaling = np.mean(errs_scaling)
    m_area_scaling = np.mean(area_scaling)
    std_err_scaling = np.std(errs_scaling)
    std_area_scaling = np.std(area_scaling)

    print(m_err_scaling)
    print(m_area_scaling)
    print(std_err_scaling)
    print(std_area_scaling)

    #np.save("2x2_kernels.npy", kernels) #D_ means the DDECs style design, the ones without it have 4 variable inputs not just 2
    #np.save("2x2_corrs.npy", corrs)
    #np.save("2x2_errs.npy", errs)
    #np.save("2x2_areas.npy", areas)