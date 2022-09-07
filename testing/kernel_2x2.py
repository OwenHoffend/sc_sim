#Fresh implementation of the 2x2 convolution kernel testing - the old ones are really cluttered
from json import load
import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1, maj_1
import matplotlib.pyplot as plt

def pcc(cs, val, precision):
    if val == 0:
        return False
    elif val == 1:
        return True

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
    for i in range(actual_precision-1):
        bit = radix_bits[i]
        if bit:
            result = np.bitwise_or(result, cs[i+1])
        else:
            result = np.bitwise_and(result, cs[i+1])
    return result

kernel = [
    0.125, 0.875,
    0.125, 0.125
]

def kernel_2x2(
    x11, x12,
    x21, x22,
    c0, c1, c2, s,
    maj = False
):
    precision = 3
    m11 = np.bitwise_and(x11, pcc([c0, c1, c2], kernel[0], precision))
    m12 = np.bitwise_and(x12, pcc([c0, c1, c2], kernel[1], precision))
    m21 = np.bitwise_and(x21, pcc([c0, c1, c2], kernel[2], precision))
    m22 = np.bitwise_and(x22, pcc([c0, c1, c2], kernel[3], precision))

    if maj:
        top = maj_1(s, m11, m21)
        bot = maj_1(s, m22, m12)
    else:
        top = mux_1(s, m11, m21)
        bot = mux_1(s, m22, m12)
    return top, bot

def kernel_2x2_maj(
    x11, x12,
    x21, x22,
    c0, c1, c2, s
):
    top, bot = kernel_2x2(
        x11, x12,
        x21, x22,
        c0, c1, c2, s,
        maj=True
    )
    return top, bot

def test_kernel_2x2():
    num_tests = 10000
    nc = 4
    nv = 4
    ptm = get_func_mat(kernel_2x2, nc+nv, 2)
    ptm_maj = get_func_mat(kernel_2x2_maj, nc+nv, 2)
    relu = SeriesCircuit([ParallelCircuit([NOT(), I(1)]), AND()]).ptm()
    A = ptm @ B_mat(2)
    K1 = A[:, 0].reshape(2**nv, 2**nc).T
    K2 = A[:, 1].reshape(2**nv, 2**nc).T
    K1_opt = opt_K_max(K1)
    K2_opt = opt_K_max(K2)
    ptm_opt = Ks_to_Mf([K1_opt, K2_opt])
    v0 = get_vin_mc0(np.array(nc*[0.5]))
    B2 = B_mat(2)
    B1 = B_mat(1)

    print(espresso_get_SOP_area(ptm, "k2x2.in", do_print=True))
    print(espresso_get_SOP_area(ptm_maj, "k2x2.in", do_print=True))
    print(espresso_get_SOP_area(ptm_opt, "k2x2.in", do_print=True))
    #opt_area_SECO(K1_opt, K2_opt, cache_file="2x2_kernel.json", print_final_espresso=True, simulated_annealing=True, sort=False)

    sccs_pre = []
    sccs_maj = []
    sccs_post = []

    errs_pre = []
    errs_maj = []
    errs_post = []

    for i in range(num_tests):
        px = np.random.rand(4)
        vin = np.kron(v0, get_vin_mc1(px))
        vout_pre = ptm.T @ vin
        vout_maj = ptm_maj.T @ vin
        vout_post = ptm_opt.T @ vin

        pout_pre_r = B1.T @ relu.T @ vout_pre
        pout_maj_r = B1.T @ relu.T @ vout_maj
        pout_post_r = B1.T @ relu.T @ vout_post

        pout_pre = B2.T @ vout_pre
        pout_maj = B2.T @ vout_maj
        pout_post = B2.T @ vout_post

        sccs_pre.append(get_corr_mat_paper(vout_pre)[0,1])
        sccs_maj.append(get_corr_mat_paper(vout_maj)[0,1])
        sccs_post.append(get_corr_mat_paper(vout_post)[0,1])

        correct = 0.5 * np.array([px[0] * kernel[0] + px[2] * kernel[2], px[1] * kernel[1] + px[3] * kernel[3]])
        correct_r = np.maximum(0, correct[0] - correct[1])
        assert np.all(np.isclose(correct, pout_pre))
        assert np.all(np.isclose(correct, pout_maj))
        assert np.all(np.isclose(correct, pout_post))

        #Correlation err
        errs_pre.append(np.abs(correct_r-pout_pre_r))
        errs_maj.append(np.abs(correct_r-pout_maj_r))
        errs_post.append(np.abs(correct_r-pout_post_r))

    print("SCC pre", np.mean(sccs_pre))
    print("SCC maj", np.mean(sccs_maj))
    print("SCC post", np.mean(sccs_post))
    print("SCC pre std", np.std(sccs_pre))
    print("SCC maj std", np.std(sccs_maj))
    print("SCC post std", np.std(sccs_post))

    print("ERR pre", np.mean(errs_pre))
    print("ERR maj", np.mean(errs_maj))
    print("ERR post", np.mean(errs_post))
    print("ERR pre std", np.std(errs_pre))
    print("ERR maj std", np.std(errs_maj))
    print("ERR post std", np.std(errs_post))