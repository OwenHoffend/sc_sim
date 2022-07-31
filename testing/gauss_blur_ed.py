import numpy as np
from sim.bitstreams import *
from sim.PTM import *
from sim.SEC import *
from sim.circuits import mux_1, maj_1, robert_cross
import matplotlib.pyplot as plt

def robert_cross_r(x11, x22, x12, x21, s):
    #Version of robert_cross with sel on the other side
    return robert_cross(s, x11, x22, x12, x21)

def gauss_blur_3x1(c0, c1, x1, x2, x3):
    return mux_1(c1, mux_1(c0, x1, x3), x2) 

def gauss_blur_3x3(
    x11, x12, x13,
    x21, x22, x23,
    x31, x32, x33,
    c0, c1, c2, c3
):
    
    a1 = gauss_blur_3x1(c0, c1, x11, x12, x13)
    a2 = gauss_blur_3x1(c0, c1, x21, x22, x23)
    a3 = gauss_blur_3x1(c0, c1, x31, x32, x33)

    return gauss_blur_3x1(c2, c3, a1, a2, a3)

def gauss_blur_4x4(
    x11, x12, x13, x14,
    x21, x22, x23, x24,
    x31, x32, x33, x34,
    x41, x42, x43, x44,
    c0, c1, c2, c3 
):

    a1 = gauss_blur_3x3(
        x11, x12, x13,
        x21, x22, x23,
        x31, x32, x33,
        c0, c1, c2, c3
    )
    a2 = gauss_blur_3x3(
        x12, x13, x14,
        x22, x23, x24,
        x32, x33, x34,
        c0, c1, c2, c3
    )
    a3 = gauss_blur_3x3(
        x21, x22, x23,
        x31, x32, x33,
        x41, x42, x43,
        c0, c1, c2, c3
    )
    a4 = gauss_blur_3x3(
        x22, x23, x24,
        x32, x33, x34,
        x42, x43, x44,
        c0, c1, c2, c3
    )
    return a1, a2, a3, a4

def test_gauss_blur_3x3():
    num_tests = 1000
    gb3_ptm = get_func_mat(gauss_blur_3x3, 13, 1)
    for _ in range(num_tests):
        px = np.random.rand(3, 3)
        gk = np.array([0.25, 0.5, 0.25])
        correct_result = gk.T @ px @ gk
        v_in = np.kron(get_vin_mc0(np.array([0.5, 0.5, 0.5, 0.5])), get_vin_mc1(px.reshape(9, )))
        result = B_mat(1).T @ gb3_ptm.T @ v_in
        assert np.isclose(correct_result, result)

def test_gauss_blur_4x4():
    num_tests = 1000
    gb4_ptm = np.load("gb4_ptm.npy")
    #gb4_ptm = get_func_mat(gauss_blur_4x4, 20, 4)
    #np.save("gb4_ptm.npy", gb4_ptm)

    rced_ptm = get_func_mat(robert_cross_r, 5, 1)
    rced_ptm = reduce_func_mat(rced_ptm, 4, 0.5)
    A = gb4_ptm @ B_mat(4)
    Ks = []
    for i in range(4):
        Ks.append(opt_K_max(A[:, i].reshape(2**4, 2**16).T))
    gb4_ptm_opt = Ks_to_Mf(Ks)

    print(espresso_get_SOP_area(gb4_ptm, "gb4.in"))
    print(espresso_get_SOP_area(gb4_ptm_opt, "gb4.in"))

    avg_corr = np.zeros((4,4))
    avg_corr_opt = np.zeros((4,4))
    B4 = B_mat(4)
    gk = np.array([0.25, 0.5, 0.25])

    unopt_err = 0.0
    opt_err = 0.0
    for i in range(num_tests):
        px = np.random.rand(4, 4)

        #Correct result computation
        c1 = gk.T @ px[0:3, 0:3] @ gk
        c2 = gk.T @ px[0:3, 1:4] @ gk
        c3 = gk.T @ px[1:4, 0:3] @ gk
        c4 = gk.T @ px[1:4, 1:4] @ gk
        rced_correct = 0.5*(np.abs(c1-c4) + np.abs(c2-c3))

        v_in = np.kron(get_vin_mc0(np.array([0.5, 0.5, 0.5, 0.5])), get_vin_mc1(px.reshape(16, )))
        result_ptv = gb4_ptm.T @ v_in
        result_ptv_opt = gb4_ptm_opt.T @ v_in
        rced_out_ptv = rced_ptm.T @ result_ptv
        rced_out_ptv_opt = rced_ptm.T @ result_ptv_opt

        unopt_err += np.abs(rced_out_ptv[1] - rced_correct)
        opt_err += np.abs(rced_out_ptv_opt[1] - rced_correct)

        result_pout = B4.T @ result_ptv
        result_pout_opt = B4.T @ result_ptv_opt
        assert np.all(np.isclose(result_pout, result_pout_opt))

        cout = get_corr_mat_paper(result_ptv)
        cout_opt = get_corr_mat_paper(result_ptv_opt)
        avg_corr += cout
        avg_corr_opt += cout_opt
    avg_corr /= num_tests
    avg_corr_opt /= num_tests
    unopt_err /= num_tests
    opt_err /= num_tests
    print(avg_corr)
    print(avg_corr_opt)
    print(unopt_err)
    print(opt_err)